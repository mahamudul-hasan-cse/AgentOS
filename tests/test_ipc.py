import asyncio

import pytest
from fastapi.testclient import TestClient

from kernel.drivers.base import LLMDriver
from kernel.ipc import Blackboard, MessageQueue
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    """Echoes its prompt back with a marker so we can trace which text flowed
    where. Because the Writer's prompt embeds the blackboard findings verbatim,
    echoing proves the findings actually reached the Writer."""

    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"FAKE_LLM::{prompt}"


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path) -> SyscallDispatcher:
    pm = PageManager(
        ram_budget_tokens=100, policy="fifo", chroma_path=str(tmp_path / "chroma_db")
    )
    return SyscallDispatcher(page_manager=pm, driver_registry={"fake": FakeDriver})


def test_direct_message_send_and_receive():
    async def scenario():
        mq = MessageQueue()
        await mq.send(to_agent="bob", from_agent="alice", content="hello bob")
        received = await mq.receive("bob", timeout=1.0)
        return received

    message = run(scenario())
    assert message is not None
    assert message.from_agent == "alice"
    assert message.to_agent == "bob"
    assert message.content == "hello bob"


def test_ipc_receive_times_out_when_inbox_empty():
    async def scenario():
        mq = MessageQueue()
        return await mq.receive("nobody", timeout=0.05)

    assert run(scenario()) is None


def test_blackboard_write_then_read_round_trips():
    async def scenario():
        bb = Blackboard()
        await bb.write("findings", {"fact": 42}, agent_id="researcher")
        value = await bb.read("findings")
        snapshot = await bb.snapshot()
        return value, snapshot

    value, snapshot = run(scenario())
    assert value == {"fact": 42}
    assert snapshot == {"findings": {"fact": 42}}


def test_ipc_syscalls_round_trip_through_dispatcher(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        send = await disp.dispatch(
            "alice", SyscallType.IPC_SEND, to_agent="bob", content="ping"
        )
        recv = await disp.dispatch("bob", SyscallType.IPC_RECV, timeout=1.0)
        return disp, send, recv

    disp, send, recv = run(scenario())
    assert send.status == SyscallStatus.SUCCESS
    assert recv.status == SyscallStatus.SUCCESS
    assert recv.result["message"]["content"] == "ping"
    assert recv.result["message"]["from_agent"] == "alice"
    # both IPC syscalls are captured in the trace
    logged_types = {s.type for s in disp.log}
    assert SyscallType.IPC_SEND in logged_types
    assert SyscallType.IPC_RECV in logged_types


def test_collaborate_flow_writer_incorporates_blackboard_findings(tmp_path):
    import api.main as m

    # repoint the app dispatcher at an offline FakeDriver + scratch chroma dir
    m.dispatcher = make_dispatcher(tmp_path)
    m.page_manager = m.dispatcher.page_manager
    client = TestClient(m.app)

    resp = client.post("/agents/collaborate", json={"topic": "photosynthesis", "driver": "fake"})
    assert resp.status_code == 200
    body = resp.json()

    # the researcher's findings landed on the blackboard (intermediate state)
    findings = body["blackboard"]["research_findings"]
    assert "photosynthesis" in findings

    # the writer's final output actually incorporates what the researcher wrote.
    # the only path for `findings` to reach the writer's echoed output is:
    # researcher -> blackboard -> writer's prompt -> FakeDriver echo.
    assert findings in body["final_output"]

    # and the whole collaboration is visible in the syscall trace
    logged_types = {s.type for s in m.dispatcher.log}
    assert SyscallType.LLM_CALL in logged_types
    assert SyscallType.BLACKBOARD_WRITE in logged_types
    assert SyscallType.BLACKBOARD_READ in logged_types
