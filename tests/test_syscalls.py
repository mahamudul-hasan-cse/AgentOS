import asyncio
import shutil
import time
from unittest.mock import MagicMock

import pytest
from groq import APITimeoutError

from kernel.drivers.base import DriverConnectionError, LLMDriver
from kernel.drivers.groq_driver import GroqDriver
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    """Offline stand-in for a cloud driver so syscall tests don't hit the network."""

    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


class TimeoutDriver(LLMDriver):
    """Simulates a cloud driver that hit its HTTP deadline."""

    name = "groq"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        raise DriverConnectionError("Request timed out")


class FallbackDriver(LLMDriver):
    name = "ollama"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"fallback: {prompt}"


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


def test_unknown_tool_call_logs_not_implemented_without_crashing(dispatcher):
    # TOOL_CALL is implemented for kernel-approved tools only. Unknown tools
    # remain NOT_IMPLEMENTED and are logged without crashing the dispatcher.
    syscall = run(dispatcher.dispatch("agent-3", SyscallType.TOOL_CALL, tool="search"))

    assert syscall.status == SyscallStatus.NOT_IMPLEMENTED
    assert syscall.result["error_type"] == "NotImplementedError"

    logged = dispatcher.log[0]
    assert logged.type == SyscallType.TOOL_CALL
    assert logged.status == SyscallStatus.NOT_IMPLEMENTED

    # the dispatcher is still fully usable after an unknown tool syscall
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


def test_groq_driver_timeout_maps_to_connection_error():
    driver = GroqDriver()
    driver._client = MagicMock()
    driver._client.chat.completions.create.side_effect = APITimeoutError("request timed out")

    with pytest.raises(DriverConnectionError):
        run(driver.generate("hello", timeout=1))


def test_llm_call_falls_back_to_ollama_when_primary_times_out(tmp_path):
    async def scenario():
        pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
        disp = SyscallDispatcher(
            page_manager=pm,
            driver_registry={"groq": TimeoutDriver, "ollama": FallbackDriver},
        )
        return await disp.dispatch(
            "agent-timeout", SyscallType.LLM_CALL, prompt="still here?", driver="groq"
        )

    syscall = run(scenario())
    assert syscall.status == SyscallStatus.SUCCESS
    assert syscall.result == {"driver_used": "ollama", "text": "fallback: still here?"}


def test_hung_sandbox_does_not_stall_concurrent_syscalls(dispatcher):
    async def scenario():
        sandbox_task = asyncio.create_task(
            dispatcher.dispatch(
                "agent-sandbox",
                SyscallType.TOOL_CALL,
                tool="python_sandbox",
                code="while True:\n    pass\n",
                timeout_seconds=0.3,
            )
        )
        await asyncio.sleep(0.05)
        llm_started = time.perf_counter()
        llm = await dispatcher.dispatch(
            "agent-live", SyscallType.LLM_CALL, prompt="concurrent", driver="fake"
        )
        llm_elapsed = time.perf_counter() - llm_started
        sandbox = await sandbox_task
        return llm, llm_elapsed, sandbox

    llm, llm_elapsed, sandbox = run(scenario())

    assert llm.status == SyscallStatus.SUCCESS
    assert llm_elapsed < 0.25, "LLM_CALL should not wait for the sandbox thread to finish"
    assert sandbox.status == SyscallStatus.SUCCESS
    assert sandbox.result["timeout"] is True
    assert sandbox.result["timeout_kill"]["pid"]
