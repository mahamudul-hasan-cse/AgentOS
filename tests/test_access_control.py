import asyncio

import pytest

from kernel.access_control import (
    AccessControl,
    AgentPrivilege,
    ProviderPool,
    ResourceManager,
)
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


class SlowDriver(LLMDriver):
    """Holds its provider slot for a beat so concurrent requests actually
    overlap, exercising the resource manager under contention."""

    name = "slow"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0.1)
        return f"slow: {prompt}"


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path, **kwargs) -> SyscallDispatcher:
    pm = PageManager(
        ram_budget_tokens=100, policy="fifo", chroma_path=str(tmp_path / "chroma_db")
    )
    return SyscallDispatcher(
        page_manager=pm, driver_registry={"fake": FakeDriver}, **kwargs
    )


# --- Access control ------------------------------------------------------


def test_user_cannot_read_another_agents_memory(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        # victim (USER) writes a page into its own memory
        await disp.dispatch(
            "victim", SyscallType.MEM_WRITE, page_id="secret", content="classified data", token_count=10
        )
        # attacker (USER, default) tries to read victim's memory
        return disp, await disp.dispatch(
            "attacker", SyscallType.MEM_READ, query_text="classified", target_agent_id="victim"
        )

    disp, syscall = run(scenario())
    assert syscall.status == SyscallStatus.PERMISSION_DENIED
    assert syscall.result["error_type"] == "PermissionDenied"
    # it was logged, not crashed
    assert disp.log[-1].status == SyscallStatus.PERMISSION_DENIED


def test_kernel_agent_can_read_another_agents_memory(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.acl.registry.register("root", AgentPrivilege.KERNEL)
        await disp.dispatch(
            "victim", SyscallType.MEM_WRITE, page_id="secret",
            content="the scheduler dispatches agent processes", token_count=10,
        )
        return await disp.dispatch(
            "root", SyscallType.MEM_READ,
            query_text="how does the scheduler dispatch", target_agent_id="victim",
        )

    syscall = run(scenario())
    assert syscall.status == SyscallStatus.SUCCESS
    assert syscall.result["page"]["page_id"] == "secret"


def test_user_can_still_access_own_memory(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        await disp.dispatch(
            "self", SyscallType.MEM_WRITE, page_id="note", content="my own note", token_count=10
        )
        # no target_agent_id -> own memory, allowed for USER
        return await disp.dispatch("self", SyscallType.MEM_READ, query_text="my own note")

    syscall = run(scenario())
    assert syscall.status == SyscallStatus.SUCCESS


def test_enforce_denies_spawn_agent_for_user():
    acl = AccessControl()
    with pytest.raises(Exception):
        acl.enforce("someone", SyscallType.SPAWN_AGENT)


# --- Resource manager / Banker's Algorithm -------------------------------


def test_bankers_algorithm_refuses_unsafe_grant():
    """Even when raw capacity remains, a grant that could deadlock is refused."""

    async def scenario():
        rm = ResourceManager(capacities={"groq": 10})
        # A holds 5 with a max claim of 10 -> could still finish (needs 5, 5 free)
        a_ok = await rm.request("A", "groq", units=5, max_claim=10)
        # B now wants 5 with a max claim of 10. Capacity has 5 free, but granting
        # leaves A needing 5 and B needing 5 with 0 available -> unsafe.
        b_ok = await rm.request("B", "groq", units=5, max_claim=10)
        return rm, a_ok, b_ok

    rm, a_ok, b_ok = run(scenario())
    assert a_ok is True
    assert b_ok is False  # refused to avoid an unsafe (deadlock-prone) state
    # B's tentative allocation was rolled back
    assert rm.state()["groq"]["allocated"] == 5


def test_concurrent_requests_never_exceed_capacity_and_fall_back(tmp_path):
    """Fire more concurrent LLM_CALLs than one provider can hold; confirm the
    resource manager never over-allocates it and overflow falls back."""

    CAPACITY = 2
    N = 6

    async def scenario():
        pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
        rm = ResourceManager(capacities={"slow": CAPACITY, "ollama": 30})
        disp = SyscallDispatcher(
            page_manager=pm,
            driver_registry={"slow": SlowDriver, "ollama": SlowDriver},
            resource_manager=rm,
        )
        results = await asyncio.gather(
            *(
                disp.dispatch(f"agent-{i}", SyscallType.LLM_CALL, prompt="go", driver="slow")
                for i in range(N)
            )
        )
        return rm, results

    rm, results = run(scenario())

    # every request completed successfully (served by 'slow' or the 'ollama' fallback)
    assert all(s.status == SyscallStatus.SUCCESS for s in results)

    state = rm.state()
    # the crucial safety invariant: 'slow' was never allocated beyond its capacity
    assert state["slow"]["peak_allocated"] <= CAPACITY
    # overflow was absorbed by the fallback provider rather than over-allocating
    assert state["ollama"]["peak_allocated"] >= 1
    # both pools ended in a safe state with everything released
    assert state["slow"]["allocated"] == 0
    assert state["ollama"]["allocated"] == 0
    assert state["slow"]["safe"] is True
