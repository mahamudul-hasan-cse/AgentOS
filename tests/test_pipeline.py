import asyncio

from agents.pipeline import PipelineRunner
from kernel.access_control import AccessControl, QuotaManager
from kernel.drivers.base import DriverConnectionError, LLMDriver
from kernel.filesystem import SemanticFS
from kernel.memory import PageManager
from kernel.sandbox import run_python_sandbox
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class PipelineDriver(LLMDriver):
    name = "ollama"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        if "Research this implementation task" in prompt:
            return "- Build a pure Python function.\n- Print a deterministic demo value."
        if "Generate a complete, runnable Python 3 script" in prompt:
            return """```python
def add(a, b):
    return a + b

print(add(2, 3))
```"""
        if "Write a final human-readable summary report" in prompt:
            return "Pipeline completed. The generated code ran and printed the expected demo output."
        return "ok"


class FailingCodeDriver(PipelineDriver):
    async def generate(self, prompt: str, **kwargs) -> str:
        if "Generate a complete, runnable Python 3 script" in prompt:
            return "raise SystemExit(3)\n"
        return await super().generate(prompt, **kwargs)


class GroqDownDriver(LLMDriver):
    name = "groq"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        raise DriverConnectionError("groq unavailable in test")


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path, drivers=None):
    acl = AccessControl()
    pm = PageManager(chroma_path=str(tmp_path / "mem_chroma"))
    fs = SemanticFS(
        access_control=acl,
        fs_root=str(tmp_path / "fs_root"),
        chroma_path=str(tmp_path / "fs_chroma"),
    )
    return SyscallDispatcher(
        access_control=acl,
        filesystem=fs,
        page_manager=pm,
        quota_manager=QuotaManager(window_seconds=60),
        driver_registry=drivers or {"ollama": PipelineDriver},
        record_state=False,
    )


def find_node(tree, pid):
    if tree.get("pid") == pid:
        return tree
    for child in tree.get("children", []):
        found = find_node(child, pid)
        if found:
            return found
    return None


def test_full_pipeline_run_creates_process_tree_under_coordinator(tmp_path):
    disp = make_dispatcher(tmp_path)
    result = run(PipelineRunner(disp).run("build an add function", driver="ollama"))

    coordinator = find_node(result["process_tree"], result["coordinator_id"])
    assert coordinator is not None
    children = {child["pid"] for child in coordinator["children"]}
    expected = {stage["agent_id"] for stage in result["stages"]}
    assert expected.issubset(children)
    assert result["status"] == "completed"


def test_tester_executes_generated_code_and_captures_real_pass_fail(tmp_path):
    passing = make_dispatcher(tmp_path / "passing")
    pass_result = run(PipelineRunner(passing).run("build an add function", driver="ollama"))
    assert pass_result["tester"]["passed"] is True
    assert pass_result["tester"]["exit_code"] == 0
    assert pass_result["tester"]["stdout"].strip() == "5"

    failing = make_dispatcher(
        tmp_path / "failing", drivers={"ollama": FailingCodeDriver}
    )
    fail_result = run(PipelineRunner(failing).run("build a failing script", driver="ollama"))
    assert fail_result["tester"]["passed"] is False
    assert fail_result["tester"]["exit_code"] == 3


def test_quota_exhaustion_is_visible_and_pipeline_continues(tmp_path):
    disp = make_dispatcher(tmp_path)
    result = run(PipelineRunner(disp).run("build an add function", driver="ollama"))

    coder = next(stage for stage in result["stages"] if stage["stage"] == "coder")
    assert coder["status"] == "success"
    assert coder["quota_events"]
    assert any(s.status == SyscallStatus.QUOTA_EXCEEDED for s in disp.log)
    assert result["status"] == "completed"
    assert result["final_report"]


def test_pipeline_trace_is_coherent_and_attributed_to_stage_agents(tmp_path):
    disp = make_dispatcher(tmp_path)
    result = run(PipelineRunner(disp).run("build an add function", driver="ollama"))
    stage_ids = {stage["agent_id"] for stage in result["stages"]}
    log_by_agent = {agent_id: [] for agent_id in stage_ids}
    for syscall in disp.log:
        if syscall.agent_id in log_by_agent:
            log_by_agent[syscall.agent_id].append(syscall.type)

    assert SyscallType.SPAWN_AGENT in [s.type for s in disp.log]
    assert SyscallType.LLM_CALL in log_by_agent[next(s["agent_id"] for s in result["stages"] if s["stage"] == "researcher")]
    assert SyscallType.FILE_WRITE in log_by_agent[next(s["agent_id"] for s in result["stages"] if s["stage"] == "coder")]
    assert SyscallType.FILE_READ in log_by_agent[next(s["agent_id"] for s in result["stages"] if s["stage"] == "tester")]
    assert SyscallType.TOOL_CALL in log_by_agent[next(s["agent_id"] for s in result["stages"] if s["stage"] == "tester")]
    assert SyscallType.BLACKBOARD_READ in log_by_agent[next(s["agent_id"] for s in result["stages"] if s["stage"] == "writer")]


def test_groq_driver_fallback_mid_pipeline_does_not_break_run(tmp_path):
    disp = make_dispatcher(
        tmp_path,
        drivers={"groq": GroqDownDriver, "ollama": PipelineDriver},
    )
    result = run(PipelineRunner(disp).run("build an add function", driver="groq"))

    assert result["status"] == "completed"
    llm_stages = [stage for stage in result["stages"] if stage["stage"] in {"researcher", "coder", "writer"}]
    assert all(stage["driver_used"] == "ollama" for stage in llm_stages)
    assert result["tester"]["passed"] is True


def test_python_sandbox_blocks_dangerous_imports_and_calls():
    blocked = [
        "import os\nos.system('echo nope')\n",
        "import subprocess\nsubprocess.run(['python', '--version'])\n",
        "eval('1 + 1')\n",
        "exec('print(1)')\n",
        "__import__('os')\n",
        "open('outside.txt', 'w').write('nope')\n",
        "import socket\n",
        "import urllib.request\n",
        "import requests\n",
    ]

    for code in blocked:
        result = run_python_sandbox(code)
        assert result["rejected"] is True
        assert result["passed"] is False


def test_python_sandbox_timeout_kills_spawned_process():
    result = run_python_sandbox("while True:\n    pass\n", timeout_seconds=0.2)

    assert result["timeout"] is True
    assert result["passed"] is False
    assert "timeout_kill" in result
    assert result["timeout_kill"]["pid"]
