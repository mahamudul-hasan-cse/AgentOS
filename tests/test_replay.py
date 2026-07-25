import asyncio

from kernel.access_control import AccessControl, AgentPrivilege, QuotaManager
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.replay import StateRecorder
from kernel.scheduler import Process
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path, interval=5, max_snapshots=200, **kwargs) -> SyscallDispatcher:
    pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
    disp = SyscallDispatcher(
        page_manager=pm, driver_registry={"fake": FakeDriver}, record_state=False, **kwargs
    )
    disp.recorder = StateRecorder(disp, interval=interval, max_snapshots=max_snapshots)
    return disp


async def _write_pages(disp, agent, count, start=0):
    for i in range(start, start + count):
        await disp.dispatch(
            agent,
            SyscallType.MEM_WRITE,
            page_id=f"pg-{i}",
            content=f"content {i}",
            token_count=5,
        )


def test_snapshots_captured_at_configured_interval(tmp_path):
    async def scenario():
        # interval of 3: a periodic snapshot every 3rd syscall
        disp = make_dispatcher(tmp_path, interval=3)
        await _write_pages(disp, "agent-1", 9)
        return disp

    disp = run(scenario())
    # 9 syscalls, none of them significant events -> 3 periodic snapshots
    assert len(disp.recorder.snapshots) == 3
    assert all("periodic snapshot" in s.label for s in disp.recorder.snapshots)
    # each snapshot records the syscall that triggered it
    assert all(s.syscall_id is not None for s in disp.recorder.snapshots)
    # ids increase monotonically
    ids = [s.snapshot_id for s in disp.recorder.snapshots]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    # state was actually captured
    assert disp.recorder.snapshots[-1].memory["agent-1"]["ram_pages"]


def test_termination_triggers_snapshot_with_label(tmp_path):
    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        # large interval so only the significant event can trigger a snapshot
        disp = make_dispatcher(tmp_path, interval=1000, access_control=acl)
        disp.scheduler.add_process(Process(pid="P2", arrival_time=0, estimated_burst=5))
        term = await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="P2")
        return disp, term

    disp, term = run(scenario())
    assert term.status == SyscallStatus.SUCCESS
    assert len(disp.recorder.snapshots) == 1
    snap = disp.recorder.snapshots[0]
    assert snap.label == "P2 terminated"
    assert snap.syscall_id == term.syscall_id
    # the terminated process is gone from the captured queue
    assert "P2" not in [p["pid"] for p in snap.processes]


def test_quota_violation_triggers_labelled_snapshot(tmp_path):
    async def scenario():
        disp = make_dispatcher(
            tmp_path, interval=1000, quota_manager=QuotaManager(default_max_pages=1)
        )
        await _write_pages(disp, "agent-3", 2)  # 2nd write exceeds the quota
        return disp

    disp = run(scenario())
    labels = [s.label for s in disp.recorder.snapshots]
    assert "quota exceeded for agent-3" in labels


def test_ring_buffer_evicts_oldest_past_its_bound(tmp_path):
    async def scenario():
        # snapshot on every syscall, but keep only the newest 4
        disp = make_dispatcher(tmp_path, interval=1, max_snapshots=4)
        await _write_pages(disp, "agent-1", 10)
        return disp

    disp = run(scenario())
    assert len(disp.recorder.snapshots) == 4  # bounded, not 10
    ids = [s.snapshot_id for s in disp.recorder.snapshots]
    assert ids == [7, 8, 9, 10]  # oldest evicted, newest retained
    # evicted snapshots are genuinely gone
    assert disp.recorder.get(1) is None
    assert disp.recorder.get(10) is not None
    assert len(disp.recorder.timeline()) == 4


def test_diff_identifies_a_terminated_process(tmp_path):
    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        disp = make_dispatcher(tmp_path, interval=1000, access_control=acl)
        disp.scheduler.add_process(Process(pid="A", arrival_time=0, estimated_burst=5, state="ready"))
        disp.scheduler.add_process(Process(pid="B", arrival_time=1, estimated_burst=3, state="ready"))

        before = disp.recorder.capture("before termination")
        await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="B")
        after = disp.recorder.snapshots[-1]
        return disp, before, after

    disp, before, after = run(scenario())
    assert after.label == "B terminated"

    diff = disp.recorder.diff(before.snapshot_id, after.snapshot_id)
    removed_pids = [p["pid"] for p in diff["processes"]["removed"]]
    assert removed_pids == ["B"]
    assert diff["processes"]["added"] == []
    assert diff["from"]["snapshot_id"] == before.snapshot_id
    assert diff["to"]["snapshot_id"] == after.snapshot_id


def test_diff_reports_page_eviction_between_snapshots(tmp_path):
    async def scenario():
        # tiny RAM budget so writes force evictions to swap
        pm = PageManager(ram_budget_tokens=20, policy="fifo", chroma_path=str(tmp_path / "chroma"))
        disp = SyscallDispatcher(
            page_manager=pm, driver_registry={"fake": FakeDriver}, record_state=False
        )
        disp.recorder = StateRecorder(disp, interval=1000)

        await _write_pages(disp, "agent-m", 2)  # 2 pages x 5 tokens: fits
        before = disp.recorder.capture("before eviction")
        # write enough to push the earliest pages out of RAM
        for i in range(2, 6):
            await disp.dispatch(
                "agent-m", SyscallType.MEM_WRITE, page_id=f"pg-{i}", content="x", token_count=10
            )
        after = disp.recorder.capture("after eviction")
        return disp, before, after

    disp, before, after = run(scenario())
    diff = disp.recorder.diff(before.snapshot_id, after.snapshot_id)
    mem = diff["memory"]["agent-m"]
    assert mem["evicted_to_swap"], "expected pages to have moved from RAM to swap"
    assert mem["pages_added"], "expected newly written pages"


def test_unknown_snapshot_id_raises_keyerror(tmp_path):
    disp = make_dispatcher(tmp_path)
    snap = disp.recorder.capture("only one")
    try:
        disp.recorder.diff(snap.snapshot_id, 9999)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_recording_never_breaks_a_syscall(tmp_path):
    """A recorder that explodes must not fail the syscall it observed."""

    class BrokenRecorder(StateRecorder):
        def observe(self, syscall):
            raise RuntimeError("recorder is broken")

    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.recorder = BrokenRecorder(disp)
        return await disp.dispatch(
            "agent-x", SyscallType.MEM_WRITE, page_id="p1", content="hello", token_count=5
        )

    syscall = run(scenario())
    assert syscall.status == SyscallStatus.SUCCESS
