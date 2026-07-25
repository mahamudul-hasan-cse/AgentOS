import shutil

import pytest

from kernel.memory import PageManager

# Each page below is deliberately sized so that any 3 of the 4 pages exceed
# the 30-token RAM budget, forcing an eviction on the 4th write.
RAM_BUDGET_TOKENS = 30
PAGE_TOKENS = 10

PAGES = [
    ("page-1", "The scheduler dispatches agent processes using FCFS or round robin."),
    ("page-2", "The memory manager treats the context window as physical RAM."),
    ("page-3", "The syscall dispatcher logs every LLM_CALL and MEM_READ event."),
    ("page-4", "The IPC layer lets agents exchange messages via a blackboard."),
]


@pytest.fixture
def chroma_path(tmp_path):
    path = str(tmp_path / "chroma_db")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def fill_ram(manager: PageManager, agent_id: str) -> None:
    for page_id, content in PAGES:
        manager.write_page(agent_id, page_id, content, token_count=PAGE_TOKENS)


def test_fifo_evicts_oldest_loaded_page(chroma_path):
    manager = PageManager(ram_budget_tokens=RAM_BUDGET_TOKENS, policy="fifo", chroma_path=chroma_path)
    fill_ram(manager, "agent-fifo")

    # budget holds 3 pages; the 4th write must evict page-1, the first one loaded
    ram_ids = set(manager.ram["agent-fifo"].keys())
    assert ram_ids == {"page-2", "page-3", "page-4"}

    state = manager.state("agent-fifo")
    swapped_ids = {p["page_id"] for p in state["swapped_pages"]}
    assert swapped_ids == {"page-1"}


def test_lru_evicts_least_recently_accessed_page(chroma_path):
    manager = PageManager(ram_budget_tokens=RAM_BUDGET_TOKENS, policy="lru", chroma_path=chroma_path)

    manager.write_page("agent-lru", "page-1", PAGES[0][1], token_count=PAGE_TOKENS)
    manager.write_page("agent-lru", "page-2", PAGES[1][1], token_count=PAGE_TOKENS)
    manager.write_page("agent-lru", "page-3", PAGES[2][1], token_count=PAGE_TOKENS)

    # touch page-1 so it's no longer the least-recently-used page
    manager.read("agent-lru", PAGES[0][1])

    # page-2 is now the least-recently-accessed page and should be evicted
    manager.write_page("agent-lru", "page-4", PAGES[3][1], token_count=PAGE_TOKENS)

    ram_ids = set(manager.ram["agent-lru"].keys())
    assert ram_ids == {"page-1", "page-3", "page-4"}

    state = manager.state("agent-lru")
    swapped_ids = {p["page_id"] for p in state["swapped_pages"]}
    assert swapped_ids == {"page-2"}


def test_page_fault_retrieves_evicted_page_from_chromadb(chroma_path):
    manager = PageManager(ram_budget_tokens=RAM_BUDGET_TOKENS, policy="fifo", chroma_path=chroma_path)
    fill_ram(manager, "agent-fault")

    # page-1 was evicted to swap by the FIFO policy during fill_ram
    assert "page-1" not in manager.ram["agent-fault"]
    assert "page-1" in {p["page_id"] for p in manager.state("agent-fault")["swapped_pages"]}

    result = manager.read("agent-fault", "How does the scheduler dispatch agent processes?")

    assert result.page_fault is True
    assert result.page.page_id == "page-1"
    assert result.page.content == PAGES[0][1]

    # the page is now back in RAM
    assert "page-1" in manager.ram["agent-fault"]

    # a subsequent read for the same content is now a RAM hit, not a fault
    hit = manager.read("agent-fault", "How does the scheduler dispatch agent processes?")
    assert hit.page_fault is False
    assert hit.page.page_id == "page-1"
