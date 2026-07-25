import asyncio

from kernel.access_control import AccessControl, AgentPrivilege, QuotaManager
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path, quota_manager=None, access_control=None) -> SyscallDispatcher:
    pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
    return SyscallDispatcher(
        page_manager=pm,
        driver_registry={"fake": FakeDriver},
        quota_manager=quota_manager,
        access_control=access_control,
    )


def test_writing_pages_past_quota_returns_quota_exceeded(tmp_path):
    async def scenario():
        # quota of 3 pages for this agent
        disp = make_dispatcher(tmp_path, quota_manager=QuotaManager(default_max_pages=3))
        results = []
        for i in range(4):
            r = await disp.dispatch(
                "agent-1",
                SyscallType.MEM_WRITE,
                page_id=f"page-{i}",
                content=f"content {i}",
                token_count=5,
            )
            results.append(r)
        return disp, results

    disp, results = run(scenario())
    # first 3 writes succeed, the 4th is refused
    assert [r.status for r in results[:3]] == [SyscallStatus.SUCCESS] * 3
    assert results[3].status == SyscallStatus.QUOTA_EXCEEDED
    assert results[3].result["error_type"] == "QuotaExceeded"
    # the over-quota page was not written
    assert "page-3" not in disp.page_manager.ram["agent-1"]
    assert disp.quota_manager.usage("agent-1")["pages_used"] == 3


def test_llm_calls_past_rate_quota_return_quota_exceeded(tmp_path):
    async def scenario():
        # 3 calls per window for this agent (all calls happen within one window)
        disp = make_dispatcher(
            tmp_path, quota_manager=QuotaManager(default_max_calls_per_minute=3)
        )
        results = []
        for _ in range(5):
            r = await disp.dispatch("caller", SyscallType.LLM_CALL, prompt="hi", driver="fake")
            results.append(r)
        return results

    results = run(scenario())
    assert [r.status for r in results[:3]] == [SyscallStatus.SUCCESS] * 3
    assert results[3].status == SyscallStatus.QUOTA_EXCEEDED
    assert results[4].status == SyscallStatus.QUOTA_EXCEEDED
    assert results[3].result["error_type"] == "QuotaExceeded"


def test_only_kernel_can_change_quota(tmp_path):
    async def scenario():
        acl = AccessControl()
        disp = make_dispatcher(tmp_path, access_control=acl)

        # USER 'attacker' tries to raise 'victim' quota -> denied
        denied = await disp.dispatch(
            "attacker", SyscallType.SET_QUOTA, target_agent_id="victim", max_pages=999
        )
        victim_after_denial = disp.quota_manager.usage("victim")["max_pages"]

        # KERNEL 'root' can
        acl.registry.register("root", AgentPrivilege.KERNEL)
        allowed = await disp.dispatch(
            "root", SyscallType.SET_QUOTA, target_agent_id="victim", max_pages=50
        )
        return denied, victim_after_denial, allowed, disp.quota_manager.usage("victim")

    denied, victim_after_denial, allowed, victim_final = run(scenario())
    assert denied.status == SyscallStatus.PERMISSION_DENIED
    assert victim_after_denial != 999  # denied attempt did not change the quota
    assert allowed.status == SyscallStatus.SUCCESS
    assert victim_final["max_pages"] == 50


def test_normal_usage_under_quota_is_unaffected(tmp_path):
    async def scenario():
        # generous default quotas
        disp = make_dispatcher(tmp_path)
        w = await disp.dispatch(
            "worker", SyscallType.MEM_WRITE, page_id="p1", content="hello", token_count=5
        )
        calls = []
        for _ in range(5):
            calls.append(
                await disp.dispatch("worker", SyscallType.LLM_CALL, prompt="hi", driver="fake")
            )
        return disp, w, calls

    disp, w, calls = run(scenario())
    assert w.status == SyscallStatus.SUCCESS
    assert all(c.status == SyscallStatus.SUCCESS for c in calls)
    usage = disp.quota_manager.usage("worker")
    assert usage["pages_used"] == 1 and usage["calls_in_window"] == 5
    assert usage["pages_used"] < usage["max_pages"]
    assert usage["calls_in_window"] < usage["max_calls_per_minute"]
