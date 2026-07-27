import asyncio

import pytest

from kernel.access_control import QuotaManager
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.scheduler import INIT_PID
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType

PARENT = "parent"
CHILD = "child"


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return "ok"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def pm(tmp_path):
    return PageManager(ram_budget_tokens=1000, policy="fifo", chroma_path=str(tmp_path / "c"))


def seed(pm, agent=PARENT, n=3, tokens=10):
    for i in range(n):
        pm.write_page(agent, f"p{i}", f"parent page {i} about scheduling", token_count=tokens)


def read_content(pm, agent, page_id):
    """Read a page's content through the agent's own page table."""
    frame = pm.frame_of(agent, page_id)
    return frame.content if frame else None


# --- 1. fork shares without duplicating ----------------------------------


def test_child_sees_parent_pages_at_spawn_with_no_duplication(pm):
    seed(pm, n=3)
    frames_before = len(pm.frames)

    stats = pm.fork(PARENT, CHILD)

    # no new frames: the child's page table points at the parent's
    assert len(pm.frames) == frames_before
    assert stats["shared_pages"] == 3
    assert stats["tokens_saved"] == 30

    for i in range(3):
        pid = f"p{i}"
        # same frame object, not a copy
        assert pm.frame_of(CHILD, pid) is pm.frame_of(PARENT, pid)
        # refcount proves sharing, not mere readability
        assert pm.refcount(CHILD, pid) == 2
        assert pm.is_shared(PARENT, pid) is True
        assert read_content(pm, CHILD, pid) == read_content(pm, PARENT, pid)

    # and the child inherits residency
    assert set(pm.ram[CHILD]) == set(pm.ram[PARENT])


def test_fork_is_zero_copy_by_token_accounting(pm):
    seed(pm, n=4, tokens=25)
    pm.fork(PARENT, CHILD)
    metrics = pm.cow_metrics()
    assert metrics["frames"] == 4                 # still only 4 physical frames
    assert metrics["page_table_entries"] == 8     # but 8 page-table entries
    assert metrics["tokens_stored"] == 100
    assert metrics["tokens_naive_copy"] == 200    # naive fork would double it
    assert metrics["tokens_saved"] == 100
    assert metrics["savings_ratio"] == 0.5


# --- 2. THE core property: a child write must not disturb the parent -----


def test_child_write_does_not_alter_what_parent_reads(pm):
    seed(pm, n=2)
    pm.fork(PARENT, CHILD)
    original = read_content(pm, PARENT, "p0")
    parent_frame = pm.frame_of(PARENT, "p0")

    pm.write_page(CHILD, "p0", "CHILD OVERWROTE THIS", token_count=10)

    # the parent is untouched — content, frame identity and refcount
    assert read_content(pm, PARENT, "p0") == original
    assert pm.frame_of(PARENT, "p0") is parent_frame
    assert pm.refcount(PARENT, "p0") == 1        # child dropped its reference
    assert pm.is_shared(PARENT, "p0") is False

    # the child got a private frame with the new content
    assert read_content(pm, CHILD, "p0") == "CHILD OVERWROTE THIS"
    assert pm.frame_of(CHILD, "p0") is not parent_frame
    assert pm.refcount(CHILD, "p0") == 1
    assert pm.cow_faults == 1

    # untouched pages stay shared
    assert pm.is_shared(PARENT, "p1") is True


def test_third_sharer_is_unaffected_when_one_writes(pm):
    """With three sharers, a write by one must leave the other two sharing."""
    seed(pm, n=1)
    pm.fork(PARENT, "c1")
    pm.fork(PARENT, "c2")
    assert pm.refcount(PARENT, "p0") == 3

    pm.write_page("c1", "p0", "only c1 changed", token_count=10)

    assert read_content(pm, "c1", "p0") == "only c1 changed"
    assert read_content(pm, PARENT, "p0") == read_content(pm, "c2", "p0")
    assert pm.frame_of(PARENT, "p0") is pm.frame_of("c2", "p0")
    assert pm.refcount(PARENT, "p0") == 2  # parent + c2 still share


# --- 4. the symmetric case: parent writes after fork ---------------------


def test_parent_write_after_fork_is_privatized_without_affecting_child(pm):
    seed(pm, n=1)
    pm.fork(PARENT, CHILD)
    child_frame = pm.frame_of(CHILD, "p0")
    original = read_content(pm, CHILD, "p0")

    pm.write_page(PARENT, "p0", "PARENT OVERWROTE THIS", token_count=10)

    assert read_content(pm, CHILD, "p0") == original
    assert pm.frame_of(CHILD, "p0") is child_frame
    assert pm.refcount(CHILD, "p0") == 1
    assert read_content(pm, PARENT, "p0") == "PARENT OVERWROTE THIS"
    assert pm.frame_of(PARENT, "p0") is not child_frame


def test_writing_a_private_page_does_not_trigger_a_cow_fault(pm):
    seed(pm, n=1)
    assert pm.cow_faults == 0
    pm.write_page(PARENT, "p0", "updated in place", token_count=10)
    assert pm.cow_faults == 0          # sole owner: no copy needed
    assert len(pm.frames) == 1
    assert read_content(pm, PARENT, "p0") == "updated in place"


# --- 3. refcount on termination ------------------------------------------


def test_refcount_decrements_on_release_and_frees_only_at_zero(pm):
    seed(pm, n=2)
    pm.fork(PARENT, CHILD)
    frame_ids = set(pm.page_table[PARENT].values())
    assert all(pm.frames[f].refcount == 2 for f in frame_ids)

    # parent goes away: frames survive because the child still references them
    result = pm.release_agent(PARENT)
    assert result["pages_released"] == 2
    assert result["frames_freed"] == []
    assert sorted(result["frames_still_shared"]) == sorted(frame_ids)
    assert all(pm.frames[f].refcount == 1 for f in frame_ids)
    assert read_content(pm, CHILD, "p0") is not None   # child unaffected

    # child goes away too: now the frames are actually freed
    result = pm.release_agent(CHILD)
    assert sorted(result["frames_freed"]) == sorted(frame_ids)
    assert pm.frames == {}


def test_terminating_a_parent_leaves_the_childs_memory_readable(tmp_path):
    async def scenario():
        disp = SyscallDispatcher(
            page_manager=PageManager(ram_budget_tokens=1000, chroma_path=str(tmp_path / "c")),
            driver_registry={"fake": FakeDriver},
            record_state=False,
        )
        from kernel.access_control import AgentPrivilege

        disp.acl.registry.register("root", AgentPrivilege.KERNEL)
        disp.scheduler.ensure_init()
        disp.scheduler.spawn(PARENT, parent_pid=INIT_PID)
        await disp.dispatch(
            PARENT, SyscallType.MEM_WRITE, page_id="p0",
            content="shared knowledge about paging", token_count=20,
        )
        spawn = await disp.dispatch(PARENT, SyscallType.SPAWN_AGENT, pid=CHILD)
        term = await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid=PARENT)
        read = await disp.dispatch(CHILD, SyscallType.MEM_READ, query_text="paging")
        return disp, spawn, term, read

    disp, spawn, term, read = run(scenario())
    assert spawn.result["inherited_pages"] == 1
    assert spawn.result["tokens_shared_not_copied"] == 20
    # the parent's teardown did not free the frame - the child still holds it
    assert term.result["frames_freed"] == []
    assert len(term.result["frames_still_shared"]) == 1
    assert read.status == SyscallStatus.SUCCESS
    assert read.result["page"]["content"] == "shared knowledge about paging"


# --- 5. quota policy: enforced on PRIVATE pages --------------------------


def test_shared_pages_are_not_charged_to_the_child_quota(tmp_path):
    """DESIGN DECISION 2: forking a parent at its page limit must not
    immediately exhaust the child's quota, because sharing costs no memory."""
    pm = PageManager(ram_budget_tokens=1000, chroma_path=str(tmp_path / "c"))
    quotas = QuotaManager(default_max_pages=3)
    quotas.bind_page_manager(pm)

    for i in range(3):
        pm.write_page(PARENT, f"p{i}", f"page {i}", token_count=10)
        quotas.record_page(PARENT, f"p{i}")
    pm.fork(PARENT, CHILD)

    child = quotas.usage(CHILD)
    assert child["pages_total"] == 3     # RSS-like view sees them
    assert child["pages_shared"] == 3
    assert child["pages_used"] == 0      # ...but nothing is charged
    assert child["pages_private"] == 0
    # so the child may still create pages of its own
    assert quotas.can_write_page(CHILD, "own-1") is True


def test_cow_write_converts_a_shared_page_into_a_charged_private_one(tmp_path):
    pm = PageManager(ram_budget_tokens=1000, chroma_path=str(tmp_path / "c"))
    quotas = QuotaManager(default_max_pages=5)
    quotas.bind_page_manager(pm)

    pm.write_page(PARENT, "p0", "original", token_count=10)
    pm.fork(PARENT, CHILD)
    assert quotas.usage(CHILD)["pages_used"] == 0

    pm.write_page(CHILD, "p0", "child's own version", token_count=10)

    child = quotas.usage(CHILD)
    assert child["pages_private"] == 1
    assert child["pages_shared"] == 0
    assert child["pages_used"] == 1      # the COW copy IS charged
    assert child["quota_charged_on"].startswith("private")


def test_quota_still_enforced_on_private_pages(tmp_path):
    pm = PageManager(ram_budget_tokens=1000, chroma_path=str(tmp_path / "c"))
    quotas = QuotaManager(default_max_pages=2)
    quotas.bind_page_manager(pm)
    for i in range(2):
        pm.write_page(PARENT, f"p{i}", "x", token_count=10)
        quotas.record_page(PARENT, f"p{i}")
    assert quotas.can_write_page(PARENT, "p2") is False  # at the private limit


# --- 6. eviction is per-agent-view ---------------------------------------


def test_evicting_a_shared_page_leaves_the_other_sharer_resident(tmp_path):
    """DESIGN DECISION 1: eviction changes residency, not ownership."""
    pm = PageManager(ram_budget_tokens=30, policy="fifo", chroma_path=str(tmp_path / "c"))
    for i in range(3):
        pm.write_page(PARENT, f"p{i}", f"page {i}", token_count=10)
    pm.fork(PARENT, CHILD)
    assert "p0" in pm.ram[PARENT] and "p0" in pm.ram[CHILD]
    frame = pm.frame_of(PARENT, "p0")
    assert frame.refcount == 2

    # push the parent's RAM over budget so p0 (oldest) is evicted from IT
    pm.write_page(PARENT, "p3", "new page", token_count=10)

    assert "p0" not in pm.ram[PARENT]      # evicted from the parent's view
    assert "p0" in pm.ram[CHILD]           # ...but still resident for the child
    assert pm.frame_of(CHILD, "p0") is frame
    assert frame.refcount == 2             # eviction never changes ownership
    assert pm.frames.get(frame.frame_id) is frame  # not freed

    # the child reads it with NO page fault; the parent must fault it back in
    assert pm.read(CHILD, "page 0").page_fault is False
    assert pm.read(PARENT, "page 0").page_fault is True


def test_evicted_shared_page_is_not_deleted_from_the_other_agents_index(tmp_path):
    pm = PageManager(ram_budget_tokens=30, policy="fifo", chroma_path=str(tmp_path / "c"))
    for i in range(3):
        pm.write_page(PARENT, f"p{i}", f"page {i}", token_count=10)
    pm.fork(PARENT, CHILD)
    pm.write_page(PARENT, "p3", "new page", token_count=10)

    child_state = pm.state(CHILD)
    resident = {p["page_id"] for p in child_state["ram_pages"]}
    assert "p0" in resident
    parent_state = pm.state(PARENT)
    swapped = {p["page_id"] for p in parent_state["swapped_pages"]}
    assert "p0" in swapped


# --- 7. metrics ----------------------------------------------------------


def test_state_reports_cow_metrics(pm):
    seed(pm, n=2)
    pm.fork(PARENT, CHILD)
    pm.write_page(CHILD, "p0", "diverged", token_count=10)

    state = pm.state(CHILD)
    assert state["cow"]["pages_total"] == 2
    assert state["cow"]["pages_shared"] == 1     # p1 still shared
    assert state["cow"]["pages_private"] == 1    # p0 now private
    assert state["cow"]["cow_faults"] == 1
    assert state["cow_global"]["cow_faults"] == 1
    assert state["cow_global"]["tokens_saved"] == 10  # p1 shared by two

    for page in state["ram_pages"]:
        assert "shared" in page and "refcount" in page
