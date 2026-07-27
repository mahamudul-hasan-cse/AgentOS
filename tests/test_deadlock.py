import asyncio

from kernel.access_control import (
    AccessControl,
    AgentPrivilege,
    DeadlockDetector,
    ResourceManager,
    find_cycle,
)
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.scheduler import INIT_PID, Process, Scheduler
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return "ok"


def run(coro):
    return asyncio.run(coro)


async def build_circular_wait(rm: ResourceManager) -> None:
    """The textbook unsafe interleaving on a single pool of capacity 2.

    A and B each declare a max claim of 2 and take one unit, then each asks for
    its second. Neither can proceed and neither will release: a circular wait.
    """
    await rm.request("A", "pool", units=1, max_claim=2)
    await rm.request("B", "pool", units=1, max_claim=2)
    await rm.request("A", "pool", units=1, max_claim=2)  # blocks
    await rm.request("B", "pool", units=1, max_claim=2)  # blocks


# --- pure cycle-finding ---------------------------------------------------


def test_find_cycle_returns_the_actual_members():
    assert find_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}}) is not None
    cycle = find_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}})
    assert set(cycle) == {"a", "b", "c"}


def test_find_cycle_returns_none_for_a_dag():
    assert find_cycle({"a": {"b"}, "b": {"c"}}) is None
    assert find_cycle({}) is None


def test_find_cycle_ignores_a_branch_that_does_not_close():
    edges = {"a": {"b", "x"}, "b": {"a"}, "x": set()}
    cycle = find_cycle(edges)
    assert set(cycle) == {"a", "b"}  # only the closing pair


# --- 1. detection with avoidance disabled ---------------------------------


def test_circular_wait_is_detected_with_avoidance_disabled():
    async def scenario():
        rm = ResourceManager(capacities={"pool": 2}, avoidance_enabled=False)
        await build_circular_wait(rm)
        detector = DeadlockDetector(rm)
        return rm, detector.detect()

    rm, result = run(scenario())

    # greedy mode really did hand both units out
    assert rm.state()["pool"]["allocated"] == 2
    assert result.deadlocked is True
    assert set(result.cycle) == {"A", "B"}
    # and the graph shows the mutual wait
    edges = {(e["from"], e["to"]) for e in result.graph["edges"]}
    assert ("A", "B") in edges and ("B", "A") in edges


# --- 2. no false positives ------------------------------------------------


def test_no_false_positive_when_holders_are_not_in_a_cycle():
    async def scenario():
        # C simply holds the whole pool; D waits. D -> C is an edge, but C waits
        # on nothing, so there is no cycle.
        rm = ResourceManager(capacities={"pool": 1}, avoidance_enabled=False)
        await rm.request("C", "pool", units=1)
        await rm.request("D", "pool", units=1)  # blocks
        return DeadlockDetector(rm).detect()

    result = run(scenario())
    assert result.deadlocked is False
    assert result.cycle == []
    edges = {(e["from"], e["to"]) for e in result.graph["edges"]}
    assert ("D", "C") in edges
    assert ("C", "D") not in edges


def test_no_false_positive_when_nothing_is_waiting():
    async def scenario():
        rm = ResourceManager(capacities={"pool": 4}, avoidance_enabled=False)
        await rm.request("X", "pool", units=1)
        await rm.request("Y", "pool", units=1)
        return DeadlockDetector(rm).detect()

    result = run(scenario())
    assert result.deadlocked is False
    assert result.graph["edges"] == []


# --- 3. recovery + victim selection ---------------------------------------


def test_recovery_terminates_the_correct_victim_and_clears_the_cycle():
    async def scenario():
        rm = ResourceManager(capacities={"pool": 3}, avoidance_enabled=False)
        # BIG holds 2 units, SMALL holds 1 -> SMALL is the cheaper victim
        await rm.request("BIG", "pool", units=2, max_claim=3)
        await rm.request("SMALL", "pool", units=1, max_claim=3)
        await rm.request("BIG", "pool", units=1, max_claim=3)    # blocks
        await rm.request("SMALL", "pool", units=1, max_claim=3)  # blocks

        scheduler = Scheduler()
        scheduler.ensure_init()
        scheduler.spawn("BIG", parent_pid=INIT_PID)
        scheduler.spawn("SMALL", parent_pid=INIT_PID)

        killed = []

        async def fake_terminate(pid):
            killed.append(pid)
            scheduler.terminate(pid)

        detector = DeadlockDetector(rm, scheduler=scheduler, terminate=fake_terminate)
        before = detector.detect()
        record = await detector.recover(before.cycle)
        after = detector.detect()
        return before, record, after, killed

    before, record, after, killed = run(scenario())

    assert before.deadlocked is True
    assert set(before.cycle) == {"BIG", "SMALL"}

    # policy: fewest resources held wins the sacrifice
    assert record["victim"] == "SMALL"
    assert killed == ["SMALL"]
    assert record["held_by_victim"] == {"pool": 1}
    assert record["freed"] == {"pool": 1}

    # the cycle is genuinely gone afterwards
    assert record["recovered"] is True
    assert after.deadlocked is False
    assert after.cycle == []


def test_victim_selection_tie_breaks_on_priority_then_arrival():
    """With equal holdings the policy prefers the lower-priority process (a
    HIGHER priority number here), then the most recent arrival."""

    async def scenario():
        rm = ResourceManager(capacities={"pool": 2}, avoidance_enabled=False)
        await rm.request("P", "pool", units=1, max_claim=2)
        await rm.request("Q", "pool", units=1, max_claim=2)
        await rm.request("P", "pool", units=1, max_claim=2)
        await rm.request("Q", "pool", units=1, max_claim=2)

        scheduler = Scheduler()
        scheduler.ensure_init()
        # equal holdings; Q has the larger priority number => lower priority
        scheduler.add_process(Process(pid="P", arrival_time=0, estimated_burst=1, priority=0))
        scheduler.add_process(Process(pid="Q", arrival_time=0, estimated_burst=1, priority=5))
        return DeadlockDetector(rm, scheduler=scheduler)

    detector = run(scenario())
    assert detector.select_victim(["P", "Q"]) == "Q"


def test_recovery_is_a_noop_when_there_is_no_deadlock():
    async def scenario():
        rm = ResourceManager(capacities={"pool": 4}, avoidance_enabled=False)
        await rm.request("solo", "pool", units=1)
        return await DeadlockDetector(rm).recover()

    record = run(scenario())
    assert record["recovered"] is False
    assert record["reason"] == "no deadlock"


# --- 4. avoidance prevents the deadlock entirely --------------------------


def test_avoidance_enabled_prevents_the_deadlock_from_forming():
    """The same interleaving that deadlocks in greedy mode is refused by the
    Banker's Algorithm, so no cycle ever exists — the two strategies are
    alternatives, and this is avoidance doing its job."""

    async def scenario():
        rm = ResourceManager(capacities={"pool": 2}, avoidance_enabled=True)
        grants = [
            await rm.request("A", "pool", units=1, max_claim=2),
            await rm.request("B", "pool", units=1, max_claim=2),
            await rm.request("A", "pool", units=1, max_claim=2),
            await rm.request("B", "pool", units=1, max_claim=2),
        ]
        return rm, grants, DeadlockDetector(rm).detect()

    rm, grants, result = run(scenario())

    # B's very first request is refused: granting it would be unsafe
    assert grants[0] is True
    assert grants[1] is False
    # no cycle, because A is never blocked - it can still finish and release
    assert result.deadlocked is False
    assert result.cycle == []
    assert "A" not in result.graph["waiting"].get("pool", {})


def test_same_scenario_deadlocks_only_when_avoidance_is_off():
    """Side-by-side contrast, the demo this whole feature exists for."""

    async def scenario(enabled: bool):
        rm = ResourceManager(capacities={"pool": 2}, avoidance_enabled=enabled)
        await build_circular_wait(rm)
        return DeadlockDetector(rm).detect().deadlocked

    assert run(scenario(True)) is False    # avoidance: prevented
    assert run(scenario(False)) is True    # detection: found


def test_mode_can_be_toggled_at_runtime():
    rm = ResourceManager(capacities={"pool": 2})
    assert rm.avoidance_enabled is True
    assert rm.set_avoidance(False) is False
    assert rm.avoidance_enabled is False
    assert rm.set_avoidance(True) is True


# --- integration: through the dispatcher ----------------------------------


def test_detection_and_recovery_are_logged_as_syscalls(tmp_path):
    async def scenario():
        acl = AccessControl()
        acl.registry.register("kernel", AgentPrivilege.KERNEL)
        rm = ResourceManager(capacities={"pool": 2}, avoidance_enabled=False)
        disp = SyscallDispatcher(
            page_manager=PageManager(chroma_path=str(tmp_path / "chroma")),
            driver_registry={"fake": FakeDriver},
            access_control=acl,
            resource_manager=rm,
            record_state=False,
        )
        disp.scheduler.ensure_init()
        disp.scheduler.spawn("A", parent_pid=INIT_PID)
        disp.scheduler.spawn("B", parent_pid=INIT_PID)
        await build_circular_wait(rm)

        scan = await disp.run_deadlock_scan()
        return disp, scan

    disp, scan = run(scenario())

    assert scan["deadlocked"] is True
    assert scan["recovery"]["recovered"] is True

    logged = [s.type for s in disp.log]
    assert SyscallType.DEADLOCK_DETECT in logged
    assert SyscallType.DEADLOCK_RECOVER in logged
    # the victim was killed through the normal TERMINATE_AGENT path
    assert SyscallType.TERMINATE_AGENT in logged
    assert all(s.status == SyscallStatus.SUCCESS for s in disp.log)


def test_monitor_follows_the_mode_and_is_cancelled_on_shutdown():
    """The background scan runs exactly when avoidance is off, and never
    outlives the app."""
    from fastapi.testclient import TestClient

    import api.main as m

    with TestClient(m.app) as client:
        # default: avoidance on -> nothing to detect, so no monitor
        assert client.get("/deadlock/status").json()["monitoring"] is False

        off = client.post("/resources/mode", json={"avoidance_enabled": False}).json()
        assert off["monitoring"] is True
        assert client.get("/deadlock/status").json()["monitoring"] is True

        on = client.post("/resources/mode", json={"avoidance_enabled": True}).json()
        assert on["monitoring"] is False
        assert client.get("/deadlock/status").json()["monitoring"] is False

    # shutdown cancelled the task
    assert m.dispatcher.deadlock_detector._task is None
    # leave the shared app in its default state for other tests
    m.dispatcher.resource_manager.set_avoidance(True)


def test_detect_syscall_is_kernel_only(tmp_path):
    async def scenario():
        disp = SyscallDispatcher(
            page_manager=PageManager(chroma_path=str(tmp_path / "chroma")),
            driver_registry={"fake": FakeDriver},
            record_state=False,
        )
        return await disp.dispatch("plainuser", SyscallType.DEADLOCK_DETECT)

    syscall = run(scenario())
    assert syscall.status == SyscallStatus.PERMISSION_DENIED
