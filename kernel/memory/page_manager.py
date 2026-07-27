"""Treats the LLM context window as physical RAM measured in tokens.

Each `Page` is a chunk of an agent's conversation history. `PageManager`
tracks, per agent, which pages currently fit in the RAM budget. When a new
or reloaded page would exceed that budget, it evicts a page (per the chosen
replacement policy) to ChromaDB, which acts as swap storage. A page fault
occurs when a query references content no longer in RAM: the page-fault
handler searches the swap store by embedding similarity (this is RAG,
reframed as OS-style paging) and loads the best match back into RAM.

How semantic that similarity really is depends on the active embedding
backend (see kernel/memory/embeddings.py). With the default OllamaEmbedder it
is genuinely semantic — real learned embeddings, so a page can be retrieved by
meaning even when it shares no words with the query. If Ollama is unreachable
the kernel falls back to HashingEmbedder, where similarity degrades to shared
vocabulary only; Semantic-LRU still works, but "semantic" is then approximate.
The backend in use is logged at startup.

COPY-ON-WRITE
=============
Memory is split the way a real OS splits it:

  * a FRAME (`Frame`) is physical memory — one copy of some content, with a
    refcount of how many agents reference it;
  * a PAGE TABLE (`self.page_table[agent]`) maps an agent's page_ids to frames;
  * the RESIDENT SET (`self.ram[agent]`) is the subset currently in RAM, the
    rest having been evicted to the swap collection.

`fork(parent, child)` points the child's page table at the parent's existing
frames and bumps their refcounts. No content is duplicated, so fork is O(number
of pages) in bookkeeping and zero in content bytes. The first write through any
sharer of a shared frame triggers a COW fault: a private frame is allocated for
just that writer, and every other sharer is untouched.

DESIGN DECISION 1 — eviction is PER-AGENT-VIEW, never global
------------------------------------------------------------
Eviction changes *residency*, not *ownership*. Each agent has its own page table
and its own RAM budget, so evicting agent A's view of a shared frame moves only
A's entry from RAM to swap; B keeps its resident view and reads it with no page
fault. Refcounts are therefore untouched by eviction — a frame is freed only
when the last page-table entry referencing it goes away (termination or COW),
never because somebody's RAM filled up. A global eviction would be wrong here:
it would let one agent's memory pressure silently degrade an unrelated agent's
performance, and would make a shared page's residency depend on which agent
happened to touch it last.

DESIGN DECISION 2 — quotas are enforced on PRIVATE pages, reported on both
-------------------------------------------------------------------------
A shared page consumes no additional physical memory, so charging it to every
sharer would bill a resource that was never spent — and would make fork useless,
since a child would inherit its parent's page count and hit its quota
immediately. The quota therefore bounds an agent's PRIVATE pages (the OS notion
of unique set size); shared pages are reported but not charged. The moment an
agent writes to a shared page, COW gives it a private copy and that copy *is*
charged — exactly how Linux memory cgroups behave, where a COW page is billed
to whoever writes it. `usage()` reports pages_private (the enforced number),
pages_shared and pages_total (an RSS-like view) so the distinction is never
ambiguous.
"""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .embeddings import (
    DEFAULT_CHROMA_PATH,
    collection_name,
    embed_text,
    estimate_tokens,
    get_chroma_client,
)
from .page_manager_types import Frame, Page, ReadResult
from .replacement import select_victim

COSINE_SPACE_METADATA = {"hnsw:space": "cosine"}
#: separates agent from page in a ChromaDB document id. Views must be keyed per
#: agent because two agents can legitimately share the same page_id, and a
#: ChromaDB id is global.
DOC_ID_SEP = "::"

__all__ = ["Page", "Frame", "ReadResult", "PageManager"]


class PageManager:
    def __init__(
        self,
        ram_budget_tokens: int = 500,
        policy: str = "fifo",
        chroma_path: str = DEFAULT_CHROMA_PATH,
    ):
        self.ram_budget_tokens = ram_budget_tokens
        self.policy = policy
        self.ram: Dict[str, "OrderedDict[str, Page]"] = defaultdict(OrderedDict)
        #: agent -> {page_id: frame_id}; spans RAM *and* swap
        self.page_table: Dict[str, Dict[str, str]] = defaultdict(dict)
        #: frame_id -> Frame; the physical page store
        self.frames: Dict[str, Frame] = {}
        self._frame_counter = itertools.count(1)
        self.cow_faults = 0
        self.cow_faults_by_agent: Dict[str, int] = defaultdict(int)

        self.client = get_chroma_client(chroma_path)
        # collections are namespaced by embedding backend + dimension, so
        # switching backends never collides with a dimension-locked collection
        self.ram_collection = self.client.get_or_create_collection(
            collection_name("ram_pages"), metadata=COSINE_SPACE_METADATA
        )
        self.swap_collection = self.client.get_or_create_collection(
            collection_name("swap_pages"), metadata=COSINE_SPACE_METADATA
        )

    # --- identity helpers --------------------------------------------------

    @staticmethod
    def _doc_id(agent_id: str, page_id: str) -> str:
        return f"{agent_id}{DOC_ID_SEP}{page_id}"

    @staticmethod
    def _page_id_of(doc_id: str) -> str:
        return doc_id.split(DOC_ID_SEP, 1)[1] if DOC_ID_SEP in doc_id else doc_id

    def _new_frame(self, content: str, token_count: int, embedding: List[float]) -> Frame:
        frame = Frame(
            frame_id=f"f{next(self._frame_counter)}",
            content=content,
            token_count=token_count,
            embedding=embedding,
        )
        self.frames[frame.frame_id] = frame
        return frame

    def _release_frame(self, frame_id: Optional[str], agent_id: str) -> bool:
        """Drop one reference. Returns True if the frame was actually freed."""
        frame = self.frames.get(frame_id) if frame_id else None
        if frame is None:
            return False
        if frame.detach(agent_id) <= 0:
            del self.frames[frame_id]
            return True
        return False

    def ram_tokens(self, agent_id: str) -> int:
        return sum(p.token_count for p in self.ram[agent_id].values())

    # --- writes (incl. the copy-on-write fault) ----------------------------

    def write_page(
        self,
        agent_id: str,
        page_id: str,
        content: str,
        token_count: Optional[int] = None,
        policy: Optional[str] = None,
    ) -> Tuple[Page, List[str]]:
        """Add or overwrite a page in an agent's memory, evicting if needed.

        Overwriting a page whose frame is SHARED triggers a copy-on-write fault:
        the writer gets a fresh private frame and the other sharers keep the old
        one, unchanged."""
        token_count = token_count if token_count is not None else estimate_tokens(content)
        if token_count > self.ram_budget_tokens:
            raise ValueError(
                f"page '{page_id}' ({token_count} tokens) exceeds the RAM budget "
                f"({self.ram_budget_tokens} tokens) on its own"
            )

        embedding = embed_text(content)
        existing_frame_id = self.page_table[agent_id].get(page_id)
        existing = self.frames.get(existing_frame_id) if existing_frame_id else None

        if existing is not None and existing.shared:
            # COPY-ON-WRITE FAULT: never mutate a frame other agents can see.
            frame = self._copy_on_write(agent_id, page_id, content, token_count, embedding)
        elif existing is not None:
            # sole owner: safe to update the frame in place
            existing.content = content
            existing.token_count = token_count
            existing.embedding = embedding
            frame = existing
        else:
            frame = self._new_frame(content, token_count, embedding)
            frame.attach(agent_id)

        self.page_table[agent_id][page_id] = frame.frame_id
        page = Page(
            page_id=page_id,
            agent_id=agent_id,
            content=frame.content,
            token_count=frame.token_count,
            embedding=frame.embedding,
            frame_id=frame.frame_id,
        )

        already_resident = page_id in self.ram[agent_id]
        incoming = 0 if already_resident else token_count
        evicted = self._make_room(
            agent_id, incoming, query_embedding=embedding, policy=policy
        )
        self._load_into_ram(page)
        return page, evicted

    def _copy_on_write(
        self,
        agent_id: str,
        page_id: str,
        content: str,
        token_count: int,
        embedding: List[float],
    ) -> Frame:
        """Split a shared frame: allocate a private one for `agent_id` only."""
        old_frame_id = self.page_table[agent_id][page_id]
        private = self._new_frame(content, token_count, embedding)
        private.attach(agent_id)
        # the writer stops referencing the shared frame; every other sharer is
        # left pointing at it, with its original content intact
        self._release_frame(old_frame_id, agent_id)
        self.cow_faults += 1
        self.cow_faults_by_agent[agent_id] += 1
        return private

    # --- fork --------------------------------------------------------------

    def fork(self, parent_id: str, child_id: str) -> Dict[str, Any]:
        """Point `child_id`'s page table at every page `parent_id` owns.

        Zero content is copied — refcounts are incremented and the child gets
        its own page-table entries and its own ChromaDB views (needed so the
        child can search its own memory). Returns stats for the caller to log."""
        shared_pages = 0
        shared_tokens = 0
        for page_id, frame_id in list(self.page_table[parent_id].items()):
            frame = self.frames.get(frame_id)
            if frame is None:
                continue
            frame.attach(child_id)
            self.page_table[child_id][page_id] = frame_id
            shared_pages += 1
            shared_tokens += frame.token_count

            child_page = Page(
                page_id=page_id,
                agent_id=child_id,
                content=frame.content,
                token_count=frame.token_count,
                embedding=frame.embedding,
                frame_id=frame_id,
            )
            if page_id in self.ram[parent_id]:
                # inherit residency: the child starts with the same resident set
                self.ram[child_id][page_id] = child_page
                self._index(self.ram_collection, child_page)
            else:
                self._index(self.swap_collection, child_page)

        return {
            "parent_id": parent_id,
            "child_id": child_id,
            "shared_pages": shared_pages,
            "shared_tokens": shared_tokens,
            # what a naive copy-on-fork would have had to duplicate
            "tokens_saved": shared_tokens,
        }

    # --- reads -------------------------------------------------------------

    def read(
        self, agent_id: str, query_text: str, policy: Optional[str] = None
    ) -> ReadResult:
        """Simulate an agent read against its memory. Finds the page most
        relevant to `query_text` across RAM and swap. If the best match is
        already in RAM, this is a cache hit. If it currently lives only in
        swap, this is a page fault: retrieve it from ChromaDB and load it
        back into RAM, evicting another page if the budget requires it."""
        query_embedding = embed_text(query_text)

        ram_best = self._best_match(self.ram_collection, agent_id, query_embedding)
        swap_best = self._best_match(self.swap_collection, agent_id, query_embedding)

        if ram_best is None and swap_best is None:
            raise KeyError(f"agent '{agent_id}' has no pages in memory")

        if ram_best is not None and (swap_best is None or ram_best[1] <= swap_best[1]):
            page_id, _distance = ram_best
            page = self.ram[agent_id][page_id]
            page.last_accessed = time.time()
            return ReadResult(page=page, page_fault=False)

        return self._handle_page_fault(agent_id, swap_best[0], query_embedding, policy)

    def _handle_page_fault(
        self,
        agent_id: str,
        page_id: str,
        query_embedding: List[float],
        policy: Optional[str],
    ) -> ReadResult:
        doc_id = self._doc_id(agent_id, page_id)
        frame_id = self.page_table[agent_id].get(page_id)
        frame = self.frames.get(frame_id) if frame_id else None

        if frame is None:
            # frame table lost it (e.g. a store restored from disk): fall back
            # to the copy ChromaDB kept
            fetched = self.swap_collection.get(
                ids=[doc_id], include=["documents", "embeddings", "metadatas"]
            )
            content = fetched["documents"][0]
            embedding = list(fetched["embeddings"][0])
            token_count = int(fetched["metadatas"][0]["token_count"])
            frame = self._new_frame(content, token_count, embedding)
            frame.attach(agent_id)
            self.page_table[agent_id][page_id] = frame.frame_id

        page = Page(
            page_id=page_id,
            agent_id=agent_id,
            content=frame.content,
            token_count=frame.token_count,
            embedding=frame.embedding,
            frame_id=frame.frame_id,
        )

        self.swap_collection.delete(ids=[doc_id])
        evicted = self._make_room(
            agent_id, page.token_count, query_embedding=query_embedding, policy=policy
        )
        self._load_into_ram(page)

        return ReadResult(
            page=page,
            page_fault=True,
            evicted_page_id=evicted[-1] if evicted else None,
        )

    # --- residency ---------------------------------------------------------

    def _make_room(
        self,
        agent_id: str,
        incoming_tokens: int,
        query_embedding: List[float],
        policy: Optional[str] = None,
    ) -> List[str]:
        effective_policy = policy or self.policy
        evicted: List[str] = []
        while (
            self.ram_tokens(agent_id) + incoming_tokens > self.ram_budget_tokens
            and self.ram[agent_id]
        ):
            victim_id = select_victim(
                effective_policy,
                self.ram[agent_id],
                ram_collection=self.ram_collection,
                agent_id=agent_id,
                query_embedding=query_embedding,
            )
            # semantic_lru ranks ChromaDB ids, which are agent-scoped
            victim_id = self._page_id_of(victim_id)
            self._evict(agent_id, victim_id)
            evicted.append(victim_id)
        return evicted

    def _index(self, collection, page: Page) -> None:
        collection.upsert(
            ids=[self._doc_id(page.agent_id, page.page_id)],
            documents=[page.content],
            embeddings=[page.embedding],
            metadatas=[
                {
                    "agent_id": page.agent_id,
                    "page_id": page.page_id,
                    "token_count": page.token_count,
                    "frame_id": page.frame_id or "",
                }
            ],
        )

    def _load_into_ram(self, page: Page) -> None:
        page.last_accessed = time.time()
        self.ram[page.agent_id][page.page_id] = page
        self.swap_collection.delete(ids=[self._doc_id(page.agent_id, page.page_id)])
        self._index(self.ram_collection, page)

    def _evict(self, agent_id: str, page_id: str) -> Page:
        """Move ONE AGENT'S view of a page from RAM to swap.

        Per DESIGN DECISION 1 this is per-agent-view: the frame is untouched and
        its refcount unchanged, so other sharers keep their resident copies.
        Eviction is a residency change, never a deallocation."""
        page = self.ram[agent_id].pop(page_id)
        self.ram_collection.delete(ids=[self._doc_id(agent_id, page_id)])
        self._index(self.swap_collection, page)
        return page

    # --- teardown ----------------------------------------------------------

    def release_agent(self, agent_id: str) -> Dict[str, Any]:
        """Drop every page an agent owns, decrementing frame refcounts.

        A frame is only actually freed once its last referencing agent is gone,
        so a terminating parent never pulls memory out from under a live child."""
        freed_frames: List[str] = []
        retained_frames: List[str] = []
        doc_ids: List[str] = []

        for page_id, frame_id in list(self.page_table[agent_id].items()):
            doc_ids.append(self._doc_id(agent_id, page_id))
            if self._release_frame(frame_id, agent_id):
                freed_frames.append(frame_id)
            else:
                retained_frames.append(frame_id)

        if doc_ids:
            # only this agent's views disappear from the index
            try:
                self.ram_collection.delete(ids=doc_ids)
                self.swap_collection.delete(ids=doc_ids)
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass

        released = len(self.page_table[agent_id])
        self.page_table.pop(agent_id, None)
        self.ram.pop(agent_id, None)
        self.cow_faults_by_agent.pop(agent_id, None)

        return {
            "agent_id": agent_id,
            "pages_released": released,
            "frames_freed": freed_frames,
            "frames_still_shared": retained_frames,
        }

    # --- introspection -----------------------------------------------------

    @staticmethod
    def _best_match(
        collection, agent_id: str, query_embedding: List[float]
    ) -> Optional[Tuple[str, float]]:
        result = collection.query(
            query_embeddings=[query_embedding],
            where={"agent_id": agent_id},
            n_results=1,
        )
        ids = result["ids"][0]
        if not ids:
            return None
        metadata = (result.get("metadatas") or [[{}]])[0][0] or {}
        page_id = metadata.get("page_id") or PageManager._page_id_of(ids[0])
        return page_id, result["distances"][0][0]

    def frame_of(self, agent_id: str, page_id: str) -> Optional[Frame]:
        frame_id = self.page_table[agent_id].get(page_id)
        return self.frames.get(frame_id) if frame_id else None

    def refcount(self, agent_id: str, page_id: str) -> int:
        frame = self.frame_of(agent_id, page_id)
        return frame.refcount if frame else 0

    def is_shared(self, agent_id: str, page_id: str) -> bool:
        frame = self.frame_of(agent_id, page_id)
        return bool(frame and frame.shared)

    def private_page_count(self, agent_id: str) -> int:
        """Pages this agent alone owns — the number quotas are enforced on
        (DESIGN DECISION 2)."""
        return sum(
            1
            for frame_id in self.page_table[agent_id].values()
            if (self.frames.get(frame_id) or Frame("", "", 0)).refcount == 1
        )

    def cow_stats(self, agent_id: str) -> Dict[str, Any]:
        shared = private = shared_tokens = private_tokens = 0
        for frame_id in self.page_table[agent_id].values():
            frame = self.frames.get(frame_id)
            if frame is None:
                continue
            if frame.shared:
                shared += 1
                shared_tokens += frame.token_count
            else:
                private += 1
                private_tokens += frame.token_count
        return {
            "pages_total": shared + private,
            "pages_shared": shared,
            "pages_private": private,
            "tokens_shared": shared_tokens,
            "tokens_private": private_tokens,
            "cow_faults": self.cow_faults_by_agent.get(agent_id, 0),
        }

    def cow_metrics(self) -> Dict[str, Any]:
        """Global COW accounting, including what a naive copy-on-fork would
        have cost: every extra reference to a frame is one page a naive
        implementation would have duplicated."""
        total_refs = sum(f.refcount for f in self.frames.values())
        distinct = len(self.frames)
        tokens_resident = sum(f.token_count for f in self.frames.values())
        tokens_naive = sum(f.token_count * max(f.refcount, 1) for f in self.frames.values())
        shared_frames = sum(1 for f in self.frames.values() if f.shared)
        return {
            "frames": distinct,
            "page_table_entries": total_refs,
            "shared_frames": shared_frames,
            "private_frames": distinct - shared_frames,
            "cow_faults": self.cow_faults,
            "tokens_stored": tokens_resident,
            "tokens_naive_copy": tokens_naive,
            "tokens_saved": tokens_naive - tokens_resident,
            "savings_ratio": (
                round(1 - tokens_resident / tokens_naive, 4) if tokens_naive else 0.0
            ),
        }

    def state(self, agent_id: str) -> dict:
        ram_pages = list(self.ram[agent_id].values())
        resident_ids = set(self.ram[agent_id])
        swapped_pages = []
        for page_id, frame_id in self.page_table[agent_id].items():
            if page_id in resident_ids:
                continue
            frame = self.frames.get(frame_id)
            if frame is None:
                continue
            swapped_pages.append(
                {
                    "page_id": page_id,
                    "content": frame.content,
                    "token_count": frame.token_count,
                    "shared": frame.shared,
                    "refcount": frame.refcount,
                }
            )

        return {
            "agent_id": agent_id,
            "ram_budget_tokens": self.ram_budget_tokens,
            "ram_tokens_used": self.ram_tokens(agent_id),
            "ram_pages": [
                {
                    "page_id": p.page_id,
                    "content": p.content,
                    "token_count": p.token_count,
                    "last_accessed": p.last_accessed,
                    "shared": self.is_shared(agent_id, p.page_id),
                    "refcount": self.refcount(agent_id, p.page_id),
                }
                for p in ram_pages
            ],
            "swapped_pages": swapped_pages,
            "cow": self.cow_stats(agent_id),
            "cow_global": self.cow_metrics(),
        }
