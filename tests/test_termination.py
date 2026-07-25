import asyncio

from kernel.access_control import AccessControl, AgentPrivilege
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.scheduler import Process
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class SlowDriver(LLMDriver):
    """Blocks long enough that a call is reliably still in-flight when we kill it."""

    name = "slow"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(30)
        return f"slow: {prompt}"


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path, **kwargs) -> SyscallDispatcher:
    pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
    return SyscallDispatcher(
        page_manager=pm, driver_registry={"slow": SlowDriver}, **kwargs
    )


def test_terminate_marks_process_terminated_and_removes_from_queue(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        proc = Process(pid="p1", arrival_time=0, estimated_burst=5, state="ready")
        disp.scheduler.add_process(proc)
        # self-termination (caller == pid) is always allowed
        syscall = await disp.dispatch("p1", SyscallType.TERMINATE_AGENT, pid="p1")
        return disp, proc, syscall

    disp, proc, syscall = run(scenario())
    assert syscall.status == SyscallStatus.SUCCESS
    assert syscall.result["process_found"] is True
    assert proc.state == "terminated"
    assert "p1" not in [p.pid for p in disp.scheduler.queue]


def test_inflight_llm_call_is_cancelled_and_slot_released(tmp_path):
    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        disp = make_dispatcher(tmp_path, access_control=acl)

        # start a long-running LLM_CALL for agent 'worker'
        call_task = asyncio.create_task(
            disp.dispatch("worker", SyscallType.LLM_CALL, prompt="hi", driver="slow")
        )

        # wait until the call has acquired its provider slot and registered inflight
        for _ in range(200):
            await asyncio.sleep(0.01)
            if (
                "worker" in disp._inflight_tasks
                and disp.resource_manager.state()["slow"]["allocated"] == 1
            ):
                break

        inflight = disp._inflight_tasks.get("worker")
        allocated_before = disp.resource_manager.state()["slow"]["allocated"]

        # a KERNEL agent terminates the worker
        term = await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="worker")

        allocated_after = disp.resource_manager.state()["slow"]["allocated"]
        call_result = await call_task  # let the killed call settle
        return inflight, allocated_before, term, allocated_after, call_result

    inflight, allocated_before, term, allocated_after, call_result = run(scenario())

    assert allocated_before == 1  # the call really was holding a slot
    assert inflight is not None and inflight.cancelled()  # the task was cancelled
    assert term.status == SyscallStatus.SUCCESS
    assert term.result["cancelled_llm_call"] is True
    # the finally-release fired under cancellation: the slot is back
    assert allocated_after == 0
    # the killed LLM_CALL recorded a clean (non-success) outcome, not a crash
    assert call_result.status == SyscallStatus.ERROR
    assert call_result.result["error_type"] == "AgentTerminated"


def test_user_cannot_terminate_another_agent_but_kernel_can(tmp_path):
    async def scenario():
        acl = AccessControl()
        disp = make_dispatcher(tmp_path, access_control=acl)
        disp.scheduler.add_process(Process(pid="victim", arrival_time=0, estimated_burst=5))

        # USER 'attacker' tries to kill 'victim' -> denied
        denied = await disp.dispatch(
            "attacker", SyscallType.TERMINATE_AGENT, pid="victim"
        )
        # the denied attempt must NOT have terminated victim
        victim_survived = "victim" in [p.pid for p in disp.scheduler.queue]

        # a USER may still kill its OWN process
        disp.scheduler.add_process(Process(pid="attacker", arrival_time=0, estimated_burst=5))
        self_ok = await disp.dispatch(
            "attacker", SyscallType.TERMINATE_AGENT, pid="attacker"
        )

        # a KERNEL agent may kill anyone
        acl.registry.register("root", AgentPrivilege.KERNEL)
        kernel_ok = await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="victim")
        return denied, victim_survived, self_ok, kernel_ok

    denied, victim_survived, self_ok, kernel_ok = run(scenario())

    assert denied.status == SyscallStatus.PERMISSION_DENIED
    assert denied.result["error_type"] == "PermissionDenied"
    assert victim_survived is True  # denied attempt did not terminate victim

    assert self_ok.status == SyscallStatus.SUCCESS  # USER can kill its own process
    assert kernel_ok.status == SyscallStatus.SUCCESS
    assert kernel_ok.result["process_found"] is True
