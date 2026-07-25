import asyncio
import shutil

import pytest

from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    """Offline stand-in for a cloud driver so syscall tests don't hit the network."""

    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def dispatcher(tmp_path):
    pm = PageManager(
        ram_budget_tokens=30, policy="fifo", chroma_path=str(tmp_path / "chroma_db")
    )
    disp = SyscallDispatcher(page_manager=pm, driver_registry={"fake": FakeDriver})
    yield disp
    shutil.rmtree(str(tmp_path / "chroma_db"), ignore_errors=True)


def test_llm_call_appears_in_log_with_success_status(dispatcher):
    syscall = run(
        dispatcher.dispatch("agent-1", SyscallType.LLM_CALL, prompt="hello", driver="fake")
    )

    assert syscall.status == SyscallStatus.SUCCESS
    assert syscall.result == {"driver_used": "fake", "text": "echo: hello"}

    assert len(dispatcher.log) == 1
    logged = dispatcher.log[0]
    assert logged.type == SyscallType.LLM_CALL
    assert logged.agent_id == "agent-1"
    assert logged.status == SyscallStatus.SUCCESS
    assert logged.latency_ms is not None and logged.latency_ms >= 0


def test_mem_write_then_read_through_dispatcher(dispatcher):
    write = run(
        dispatcher.dispatch(
            "agent-2",
            SyscallType.MEM_WRITE,
            page_id="p1",
            content="the scheduler dispatches agent processes using round robin",
            token_count=10,
        )
    )
    assert write.status == SyscallStatus.SUCCESS
    assert write.result["page"]["page_id"] == "p1"

    read = run(
        dispatcher.dispatch(
            "agent-2", SyscallType.MEM_READ, query_text="how does the scheduler dispatch?"
        )
    )
    assert read.status == SyscallStatus.SUCCESS
    assert read.result["page"]["page_id"] == "p1"
    assert read.result["page_fault"] is False

    logged_types = [s.type for s in dispatcher.log]
    assert SyscallType.MEM_WRITE in logged_types
    assert SyscallType.MEM_READ in logged_types
    assert all(s.status == SyscallStatus.SUCCESS for s in dispatcher.log)


def test_unimplemented_syscall_logs_not_implemented_without_crashing(dispatcher):
    # agent-3 is a plain USER-level agent. Per ENOSYS-before-EPERM semantics,
    # an unimplemented syscall returns NOT_IMPLEMENTED regardless of privilege
    # (the handler-existence check runs before access control).
    syscall = run(dispatcher.dispatch("agent-3", SyscallType.TOOL_CALL, tool="search"))

    assert syscall.status == SyscallStatus.NOT_IMPLEMENTED
    assert syscall.result["error_type"] == "NotImplementedError"

    logged = dispatcher.log[0]
    assert logged.type == SyscallType.TOOL_CALL
    assert logged.status == SyscallStatus.NOT_IMPLEMENTED

    # the dispatcher is still fully usable after an unimplemented syscall
    follow_up = run(
        dispatcher.dispatch("agent-3", SyscallType.LLM_CALL, prompt="still alive?", driver="fake")
    )
    assert follow_up.status == SyscallStatus.SUCCESS
    assert len(dispatcher.log) == 2


def test_get_log_returns_most_recent_first_with_limit(dispatcher):
    run(dispatcher.dispatch("agent-4", SyscallType.LLM_CALL, prompt="one", driver="fake"))
    run(dispatcher.dispatch("agent-4", SyscallType.LLM_CALL, prompt="two", driver="fake"))
    run(dispatcher.dispatch("agent-4", SyscallType.LLM_CALL, prompt="three", driver="fake"))

    recent = dispatcher.get_log(limit=2)
    assert len(recent) == 2
    assert recent[0].args["prompt"] == "three"
    assert recent[1].args["prompt"] == "two"
