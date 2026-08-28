"""The kernel assistant: an agent that runs inside the kernel, as a process.

These tests are about the *architecture claim*, not about answer quality. The
claim is that the assistant is an ordinary process subject to the ordinary
rules: it appears in the process table, it reaches kernel state only by issuing
syscalls that get logged, access control refuses it like anyone else, quotas
bind it like anyone else, and it can be killed like anyone else.

The LLM is stubbed throughout. That is deliberate rather than a convenience:
the properties under test are all kernel-level, they must hold with no provider
configured (as in CI), and stubbing lets us assert what the model was actually
*handed* — which is where the grounding contract lives.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agents.kernel_assistant import ASSISTANT_PID, KernelAssistant
from kernel.filesystem import SemanticFS
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class StubDriver:
    """Deterministic stand-in for an LLM provider. Records every prompt so the
    grounding contract can be asserted on what the model was given."""

    name = "stub"
    prompts: list = []

    async def generate(self, prompt: str, **_):
        StubDriver.prompts.append(prompt)
        return "STUB ANSWER"


@pytest.fixture
def kernel(tmp_path):
    StubDriver.prompts = []
    chroma = str(tmp_path / "chroma")
    dispatcher = SyscallDispatcher(
        page_manager=PageManager(chroma_path=chroma),
        filesystem=SemanticFS(fs_root=str(tmp_path / "fs"), chroma_path=chroma),
        driver_registry={"stub": StubDriver},
        record_state=False,
    )
    assistant = KernelAssistant(dispatcher)
    run(assistant.register())
    return dispatcher, assistant


def run(coro):
    """Match the suite's convention: no pytest-asyncio dependency."""
    return asyncio.run(coro)


def ask(assistant, message, **kw):
    return run(assistant.answer(message, driver="stub", **kw))


# --- (1) it is a real process -------------------------------------------------


def test_assistant_is_a_real_process_visible_in_the_tree(kernel):
    """(1) Not a service beside the kernel — an entry in the process table."""
    dispatcher, assistant = kernel

    process = dispatcher.scheduler.get(ASSISTANT_PID)
    assert process is not None, "assistant must exist in the scheduler queue"
    assert process.parent_pid == "init"
    assert process.state not in ("terminated", "zombie")
    assert assistant.is_alive()

    # reachable by walking the hierarchy from init, like any other process
    tree = dispatcher.scheduler.get_tree()
    pids = []

    def walk(node):
        pids.append(node["pid"])
        for child in node["children"]:
            walk(child)

    walk(tree)
    assert ASSISTANT_PID in pids, f"assistant missing from process tree: {pids}"

    # USER privilege, and its own quota entry exists
    from kernel.access_control import AgentPrivilege

    assert dispatcher.acl.registry.privilege(ASSISTANT_PID) == AgentPrivilege.USER
    assert dispatcher.quota_manager.usage(ASSISTANT_PID)["max_calls_per_minute"] > 0


# --- (2) it reads live state through logged syscalls --------------------------


def test_asking_about_live_state_issues_syscalls_that_appear_in_the_log(kernel):
    """(2) The answer is produced by real syscalls, and they are in the trace."""
    dispatcher, assistant = kernel
    dispatcher.scheduler.spawn(pid="worker-1", parent_pid="init")

    before = len(dispatcher.log)
    reply = ask(assistant, "what processes are running right now?")

    issued = [s["type"] for s in reply["syscalls"]]
    assert SyscallType.PROC_LIST.value in issued, issued
    assert SyscallType.LLM_CALL.value in issued

    # they are in the kernel's own log, attributed to the assistant
    new_entries = dispatcher.log[before:]
    assert len(new_entries) == len(reply["syscalls"])
    assert {s.agent_id for s in new_entries} == {ASSISTANT_PID}
    logged_types = [s.type.value for s in new_entries]
    assert SyscallType.PROC_LIST.value in logged_types

    # and the PROC_LIST really returned the live table, including itself
    proc_list = next(
        s for s in new_entries if s.type == SyscallType.PROC_LIST
    )
    pids = {p["pid"] for p in proc_list.result["processes"]}
    assert {"worker-1", ASSISTANT_PID} <= pids

    # the model was handed that state rather than asked to recall it
    assert "worker-1" in StubDriver.prompts[-1]


def test_project_questions_are_grounded_in_repository_docs(kernel):
    """Project specifics come from the indexed docs via FILE_SEARCH/FILE_READ,
    not from the model's general knowledge of operating systems."""
    dispatcher, assistant = kernel
    indexed = run(assistant.index_documentation())
    assert indexed["indexed"] > 0 and not indexed["missing"]

    reply = ask(assistant, "what did the starvation benchmark find?")
    issued = [s["type"] for s in reply["syscalls"]]
    assert SyscallType.FILE_SEARCH.value in issued
    assert SyscallType.FILE_READ.value in issued

    prompt = StubDriver.prompts[-1]
    assert "RETRIEVED DOCUMENTATION" in prompt
    # a real section of this repo's docs reached the prompt
    assert "starvation" in prompt.lower()


# --- (3) access control refuses it, and it explains rather than fabricates -----


def test_reading_another_agents_memory_is_denied_and_explained(kernel):
    """(3) PERMISSION_DENIED is a feature demonstration, not a crash."""
    dispatcher, assistant = kernel
    dispatcher.scheduler.spawn(pid="researcher", parent_pid="init")
    run(dispatcher.dispatch(
        "researcher",
        SyscallType.MEM_WRITE,
        page_id="secret",
        content="researcher private notes",
        token_count=5,
    ))

    reply = ask(assistant, "what is stored in researcher's memory?")

    # it genuinely attempted the cross-agent read, and the kernel refused it
    denied = [
        s
        for s in reply["syscalls"]
        if s["type"] == SyscallType.MEM_STATE.value
        and s["status"] == SyscallStatus.PERMISSION_DENIED.value
    ]
    assert denied, f"expected a denied MEM_STATE, got {reply['syscalls']}"
    assert "researcher" in denied[0]["target"]

    # it did not crash, and still answered
    assert reply["process_alive"] is True
    assert reply["answer"]

    # the denial was surfaced to the model with the ACL explanation required,
    # and the refused content was NOT smuggled in from anywhere
    prompt = StubDriver.prompts[-1]
    assert "PERMISSION_DENIED" in prompt.upper()
    assert "USER privilege" in prompt or "USER-level" in prompt
    assert "researcher private notes" not in prompt, "refused data leaked into the prompt"


# --- (4) quotas bind it like any other agent ----------------------------------


def test_assistant_is_subject_to_quotas(kernel):
    """(4) Exhausting its LLM call quota yields QUOTA_EXCEEDED — no exemption."""
    dispatcher, assistant = kernel
    dispatcher.quota_manager.set_quota(ASSISTANT_PID, max_calls_per_minute=2)

    statuses = []
    for _ in range(4):
        reply = ask(assistant, "how many processes are running?")
        llm = [s for s in reply["syscalls"] if s["type"] == SyscallType.LLM_CALL.value]
        statuses.append(llm[-1]["status"])

    assert statuses[0] == SyscallStatus.SUCCESS.value
    assert SyscallStatus.QUOTA_EXCEEDED.value in statuses, statuses

    # the final answer reports the refusal instead of inventing an answer
    assert "QUOTA_EXCEEDED" in reply["answer"].upper()
    assert reply["syscalls"], "read syscalls should still be reported"


# --- (5) it can be killed, and the panel degrades gracefully ------------------


def test_terminating_the_assistant_degrades_gracefully(kernel):
    """(5) Killed like any process; afterwards it refuses to answer rather than
    pretending it can still read the kernel."""
    dispatcher, assistant = kernel
    assert assistant.is_alive()

    # kill it the way the shell does: a TERMINATE_AGENT syscall
    killed = run(dispatcher.dispatch(
        "kernel", SyscallType.TERMINATE_AGENT, pid=ASSISTANT_PID
    ))
    assert killed.status == SyscallStatus.SUCCESS
    assert killed.result["process_found"] is True
    assert not assistant.is_alive()

    reply = ask(assistant, "what processes are running?")
    assert reply["process_alive"] is False
    assert ASSISTANT_PID in reply["answer"]
    assert "terminated" in reply["answer"].lower()
    # crucially it issued NO syscalls: a dead process must not keep working
    assert reply["syscalls"] == []

    # and it can be brought back
    run(assistant.register())
    assert assistant.is_alive()


def test_shell_kill_path_takes_the_assistant_down_and_status_reports_it():
    """(5, end to end) `kill assistant` in the shell hits POST
    /scheduler/terminate/<pid>; the panel polls /assistant/status and must see
    the process gone rather than erroring."""
    from api.main import app, assistant as app_assistant

    with TestClient(app) as client:
        try:
            assert client.get("/assistant/status").json()["alive"] is True

            # exactly the request shell/repl.py's `kill` command issues
            killed = client.post(f"/scheduler/terminate/{ASSISTANT_PID}")
            assert killed.status_code == 200
            assert killed.json()["process_found"] is True

            status = client.get("/assistant/status").json()
            assert status["alive"] is False
            assert status["state"] in ("terminated", "zombie", None)

            # the chat endpoint degrades: a normal 200 the panel can render,
            # not a 500 that would break the UI
            chat = client.post("/assistant/chat", json={"message": "what is running?"})
            assert chat.status_code == 200
            body = chat.json()
            assert body["process_alive"] is False
            assert body["syscalls"] == []

            restarted = client.post("/assistant/restart").json()
            assert restarted["alive"] is True
        finally:
            # leave the module-level app usable for any other test
            run(app_assistant.register())


def test_health_reports_startup_state_without_waiting_for_doc_index(monkeypatch):
    """Health exposes optional startup work so doc indexing cannot look like a hang."""
    import api.main as m

    async def fast_seed():
        return None

    async def slow_index():
        await asyncio.sleep(0.05)
        return {"indexed": 0, "missing": []}

    monkeypatch.setattr(m, "_seed_memory_demo", fast_seed)
    monkeypatch.setattr(m.assistant, "index_documentation", slow_index)
    monkeypatch.setattr(m, "STARTUP_MEMORY_DEMO_TIMEOUT", 1.0)
    monkeypatch.setattr(m, "STARTUP_ASSISTANT_INDEX_TIMEOUT", 1.0)

    with TestClient(m.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["startup"]["status"] == "serving"
    assert body["startup"]["steps"]["assistant_register"]["status"] == "complete"
    assert "assistant_doc_index" in body["startup"]["steps"]
