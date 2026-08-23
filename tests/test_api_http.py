"""HTTP-level API tests via FastAPI TestClient (not unit tests against classes alone)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kernel.access_control import AccessControl, AgentPrivilege
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.replay import StateRecorder
from kernel.scheduler import Process
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType

from tests.test_pipeline import FailingCodeDriver, PipelineDriver, find_node, make_dispatcher


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


def run(coro):
    return asyncio.run(coro)


def make_replay_dispatcher(
    tmp_path,
    interval: int = 1,
    max_snapshots: int = 200,
    access_control: AccessControl | None = None,
) -> SyscallDispatcher:
    pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
    disp = SyscallDispatcher(
        page_manager=pm,
        driver_registry={"fake": FakeDriver},
        record_state=False,
        access_control=access_control or AccessControl(),
    )
    disp.recorder = StateRecorder(disp, interval=interval, max_snapshots=max_snapshots)
    return disp


@pytest.fixture
def isolated_api(monkeypatch):
    """Swap in a test dispatcher for HTTP calls, then restore the module globals."""
    import api.main as m

    saved = {
        "dispatcher": m.dispatcher,
        "page_manager": m.page_manager,
        "assistant_dispatcher": m.assistant.dispatcher,
    }

    async def noop():
        return None

    monkeypatch.setattr(m, "_seed_memory_demo", noop)

    def install(disp: SyscallDispatcher) -> None:
        m.dispatcher = disp
        m.page_manager = disp.page_manager
        m.assistant.dispatcher = disp

    yield m, install

    m.dispatcher = saved["dispatcher"]
    m.page_manager = saved["page_manager"]
    m.assistant.dispatcher = saved["assistant_dispatcher"]
    try:
        m.assistant.register()
    except Exception:  # noqa: BLE001 — best-effort restore for downstream tests
        pass


async def _seed_replay_snapshots(disp: SyscallDispatcher, count: int = 6) -> None:
    for i in range(count):
        await disp.dispatch(
            "agent-1",
            SyscallType.MEM_WRITE,
            page_id=f"pg-{i}",
            content=f"content {i}",
            token_count=5,
        )



def _assert_pipeline_process_tree_consistent(body: dict) -> None:
    """After a pipeline run, stage PIDs exist and none are stuck running."""
    tree = body["process_tree"]
    coord_id = f"{body['run_id']}_coordinator"
    coord = find_node(tree, coord_id)
    if coord is not None:
        assert coord["state"] in {"terminated", "zombie"}
    for stage in body["stages"]:
        node = find_node(tree, stage["agent_id"])
        assert node is not None, f"missing process node for {stage['agent_id']}"
        if stage["status"] in {"success", "failed"}:
            assert node["state"] in {"terminated", "zombie"}, (
                f"{stage['agent_id']} still {node['state']} after stage {stage['status']}"
            )


# --- replay HTTP -------------------------------------------------------------


def test_replay_timeline_snapshot_diff_over_http(tmp_path, isolated_api):
    m, install = isolated_api
    disp = make_replay_dispatcher(tmp_path, interval=2)
    run(_seed_replay_snapshots(disp, count=6))
    install(disp)

    with TestClient(m.app) as client:
        timeline = client.get("/replay/timeline")
        assert timeline.status_code == 200
        snapshots = timeline.json()["snapshots"]
        assert len(snapshots) >= 2
        assert all("snapshot_id" in s and "label" in s for s in snapshots)

        snap_id = snapshots[0]["snapshot_id"]
        detail = client.get(f"/replay/snapshot/{snap_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["snapshot_id"] == snap_id
        assert "processes" in body and "memory" in body
        assert "resources" in body and "quotas" in body

        other_id = snapshots[-1]["snapshot_id"]
        diff = client.get(f"/replay/diff/{snap_id}/{other_id}")
        assert diff.status_code == 200
        diff_body = diff.json()
        assert diff_body["from"]["snapshot_id"] == snap_id
        assert diff_body["to"]["snapshot_id"] == other_id
        assert "processes" in diff_body and "memory" in diff_body


def test_replay_snapshot_unknown_id_returns_404(tmp_path, isolated_api):
    m, install = isolated_api
    disp = make_replay_dispatcher(tmp_path, interval=1, max_snapshots=4)
    run(_seed_replay_snapshots(disp, count=10))  # evicts ids 1-6; 7-10 remain
    install(disp)
    retained_id = disp.recorder.snapshots[-1].snapshot_id

    with TestClient(m.app) as client:
        evicted = client.get("/replay/snapshot/1")
        assert evicted.status_code == 404
        assert "not found" in evicted.json()["detail"].lower()

        retained = client.get(f"/replay/snapshot/{retained_id}")
        assert retained.status_code == 200

        diff_missing = client.get(f"/replay/diff/{retained_id}/9999")
        assert diff_missing.status_code == 404
        assert "9999" in diff_missing.json()["detail"]


def test_replay_snapshot_after_termination_event(tmp_path, isolated_api):
    m, install = isolated_api

    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        disp = make_replay_dispatcher(tmp_path, interval=1000, access_control=acl)
        disp.scheduler.add_process(Process(pid="P2", arrival_time=0, estimated_burst=5))
        term = await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="P2")
        assert term.status == SyscallStatus.SUCCESS
        return disp

    disp = run(scenario())
    assert len(disp.recorder.snapshots) == 1
    install(disp)

    with TestClient(m.app) as client:
        timeline = client.get("/replay/timeline").json()["snapshots"]
        assert len(timeline) == 1
        snap = client.get(f"/replay/snapshot/{timeline[0]['snapshot_id']}")
        assert snap.status_code == 200
        pids = [p["pid"] for p in snap.json()["processes"]]
        assert "P2" not in pids


# --- pipeline HTTP failure paths ---------------------------------------------


def _run_pipeline(client: TestClient, topic: str = "build an add function", driver: str = "ollama"):
    return client.post("/pipeline/run", json={"topic": topic, "driver": driver})


def test_pipeline_happy_path_over_http(tmp_path, isolated_api):
    m, install = isolated_api
    disp = make_dispatcher(tmp_path)
    install(disp)

    with TestClient(m.app) as client:
        resp = _run_pipeline(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["tester"]["passed"] is True
        _assert_pipeline_process_tree_consistent(body)

        status = client.get("/pipeline/status")
        assert status.status_code == 200
        assert status.json()["status"] == "completed"


def test_pipeline_coder_parse_extract_failure_over_http(tmp_path, isolated_api):
    """Coder stage failure is returned as a structured failed run, not a 500."""
    m, install = isolated_api
    disp = make_dispatcher(tmp_path)
    install(disp)

    with TestClient(m.app) as client:
        with patch(
            "agents.pipeline.extract_python_code",
            side_effect=ValueError("generated code could not be parsed"),
        ):
            resp = _run_pipeline(client, topic="unparseable codegen task")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        coder = next(s for s in body["stages"] if s["stage"] == "coder")
        assert coder["status"] == "failed"
        assert "generated code could not be parsed" in (coder["error"] or "")
        assert any(
            e.get("stage") == "coder" and e.get("status") == "failed"
            for e in body.get("events", [])
        )
        _assert_pipeline_process_tree_consistent(body)

        status = client.get("/pipeline/status").json()
        assert status["status"] == "failed"
        assert status["current_stage"] is None or status["current_stage"] != "coder"

    # pipeline-related syscalls should all reach a terminal status
    pipeline_agents = {s["agent_id"] for s in body["stages"]} | {f"{body['run_id']}_coordinator"}
    pipeline_syscalls = [s for s in disp.log if s.agent_id in pipeline_agents]
    assert pipeline_syscalls
    assert all(s.status != SyscallStatus.PENDING for s in pipeline_syscalls)


def test_pipeline_tester_sandbox_failure_over_http(tmp_path, isolated_api):
    """Sandbox execution failure is visible on tester, run completes cleanly."""
    m, install = isolated_api
    disp = make_dispatcher(tmp_path / "failing", drivers={"ollama": FailingCodeDriver})
    install(disp)

    with TestClient(m.app) as client:
        resp = _run_pipeline(client, topic="build a failing script")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["tester"]["passed"] is False
        assert body["tester"]["exit_code"] == 3

        tester_stage = next(s for s in body["stages"] if s["stage"] == "tester")
        assert tester_stage["status"] == "success"
        assert tester_stage["produced"] == "fail"
        assert body["final_report"]

        _assert_pipeline_process_tree_consistent(body)
        assert client.get("/pipeline/status").json()["tester"]["passed"] is False


def test_pipeline_quota_exhaustion_mid_run_over_http(tmp_path, isolated_api):
    """Quota exhaustion during coder refine is visible and does not 500 or hang."""
    m, install = isolated_api
    disp = make_dispatcher(tmp_path)
    install(disp)

    with TestClient(m.app) as client:
        resp = _run_pipeline(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"

        coder = next(s for s in body["stages"] if s["stage"] == "coder")
        assert coder["status"] == "success"
        assert coder["quota_events"], "expected visible quota_events on coder stage"
        assert any(
            evt.get("status") == SyscallStatus.QUOTA_EXCEEDED.value
            for evt in coder["quota_events"]
        )

        assert body["final_report"]
        _assert_pipeline_process_tree_consistent(body)

        status = client.get("/pipeline/status").json()
        assert status["status"] == "completed"
        status_coder = next(s for s in status["stages"] if s["stage"] == "coder")
        assert status_coder["quota_events"]

    assert any(s.status == SyscallStatus.QUOTA_EXCEEDED for s in disp.log)
