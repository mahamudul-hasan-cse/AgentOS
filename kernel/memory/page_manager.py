"""Treats the LLM context window as physical RAM measured in tokens.

Each `Page` is a chunk of an agent's conversation history. `PageManager`
tracks, per agent, which pages currently fit in the RAM budget. When a new
or reloaded page would exceed that budget, it evicts a page (per the chosen
replacement policy) to ChromaDB, which acts as swap storage. A page fault
occurs when a query references content no longer in RAM: the page-fault
handler searches the swap store by embedding similarity (this is RAG,
reframed as OS-style paging) and loads the best match back into RAM.
"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

from .embeddings import DEFAULT_CHROMA_PATH, embed_text, estimate_tokens, get_chroma_client
from .page_manager_types import Page, ReadResult
from .replacement import select_victim

COSINE_SPACE_METADATA = {"hnsw:space": "cosine"}

__all__ = ["Page", "ReadResult", "PageManager"]


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

        self.client = get_chroma_client(chroma_path)
        self.ram_collection = self.client.get_or_create_collection(
            "ram_pages", metadata=COSINE_SPACE_METADATA
        )
        self.swap_collection = self.client.get_or_create_collection(
            "swap_pages", metadata=COSINE_SPACE_METADATA
        )

    def ram_tokens(self, agent_id: str) -> int:
        return sum(p.token_count for p in self.ram[agent_id].values())

    def write_page(
        self,
        agent_id: str,
        page_id: str,
        content: str,
        token_count: Optional[int] = None,
        policy: Optional[str] = None,
    ) -> Tuple[Page, List[str]]:
        """Add a page to an agent's memory, evicting pages if it doesn't fit."""
        token_count = token_count if token_count is not None else estimate_tokens(content)
        if token_count > self.ram_budget_tokens:
            raise ValueError(
                f"page '{page_id}' ({token_count} tokens) exceeds the RAM budget "
                f"({self.ram_budget_tokens} tokens) on its own"
            )

        embedding = embed_text(content)
        page = Page(
            page_id=page_id,
            agent_id=agent_id,
            content=content,
            token_count=token_count,
            embedding=embedding,
        )

        evicted = self._make_room(
            agent_id, token_count, query_embedding=embedding, policy=policy
        )
        self._load_into_ram(page)
        return page, evicted

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
        fetched = self.swap_collection.get(
            ids=[page_id], include=["documents", "embeddings", "metadatas"]
        )
        content = fetched["documents"][0]
        embedding = list(fetched["embeddings"][0])
        metadata = fetched["metadatas"][0]
        page = Page(
            page_id=page_id,
            agent_id=agent_id,
            content=content,
            token_count=int(metadata["token_count"]),
            embedding=embedding,
        )

        self.swap_collection.delete(ids=[page_id])
        evicted = self._make_room(
            agent_id, page.token_count, query_embedding=query_embedding, policy=policy
        )
        self._load_into_ram(page)

        return ReadResult(
            page=page,
            page_fault=True,
            evicted_page_id=evicted[-1] if evicted else None,
        )

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
            self._evict(agent_id, victim_id)
            evicted.append(victim_id)
        return evicted

    def _load_into_ram(self, page: Page) -> None:
        page.last_accessed = time.time()
        self.ram[page.agent_id][page.page_id] = page
        self.ram_collection.upsert(
            ids=[page.page_id],
            documents=[page.content],
            embeddings=[page.embedding],
            metadatas=[{"agent_id": page.agent_id, "token_count": page.token_count}],
        )

    def _evict(self, agent_id: str, page_id: str) -> Page:
        page = self.ram[agent_id].pop(page_id)
        self.ram_collection.delete(ids=[page_id])
        self.swap_collection.upsert(
            ids=[page.page_id],
            documents=[page.content],
            embeddings=[page.embedding],
            metadatas=[{"agent_id": page.agent_id, "token_count": page.token_count}],
        )
        return page

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
        return ids[0], result["distances"][0][0]

    def state(self, agent_id: str) -> dict:
        ram_pages = list(self.ram[agent_id].values())
        swapped = self.swap_collection.get(
            where={"agent_id": agent_id}, include=["documents", "metadatas"]
        )
        swapped_pages = [
            {"page_id": pid, "content": doc, "token_count": meta.get("token_count")}
            for pid, doc, meta in zip(
                swapped["ids"], swapped["documents"], swapped["metadatas"]
            )
        ]
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
                }
                for p in ram_pages
            ],
            "swapped_pages": swapped_pages,
        }
