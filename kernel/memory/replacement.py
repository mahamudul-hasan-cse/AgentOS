"""Page replacement policies used by PageManager when RAM is full.

Each policy returns the `page_id` of the page to evict. FIFO and LRU decide
using only the in-memory `Page` bookkeeping; SemanticLRU delegates the
similarity ranking to ChromaDB's own vector search (cosine space) instead of
computing cosine similarity by hand in Python.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

from .page_manager_types import Page

POLICY_NAMES = ("fifo", "lru", "semantic_lru")


def fifo_evict(pages: "OrderedDict[str, Page]") -> str:
    """Evict the oldest-loaded page (first key in insertion order)."""
    return next(iter(pages))


def lru_evict(pages: "OrderedDict[str, Page]") -> str:
    """Evict the least-recently-accessed page."""
    return min(pages.values(), key=lambda p: p.last_accessed).page_id


def semantic_lru_evict(
    pages: "OrderedDict[str, Page]",
    ram_collection,
    agent_id: str,
    query_embedding: List[float],
) -> str:
    """Evict the page with the lowest cosine similarity to the current query,
    i.e. the page ChromaDB's own similarity search ranks farthest away among
    the agent's currently-loaded pages."""
    if not pages:
        raise ValueError("no pages in RAM to evict")

    result = ram_collection.query(
        query_embeddings=[query_embedding],
        where={"agent_id": agent_id},
        n_results=len(pages),
    )
    ranked_ids = result["ids"][0]
    if not ranked_ids:
        # fall back to FIFO if the RAM index is out of sync for some reason
        return fifo_evict(pages)
    return ranked_ids[-1]  # farthest from the query = least similar


def select_victim(
    policy: str,
    pages: "OrderedDict[str, Page]",
    *,
    ram_collection=None,
    agent_id: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
) -> str:
    if not pages:
        raise ValueError("no pages in RAM to evict")

    if policy == "fifo":
        return fifo_evict(pages)
    if policy == "lru":
        return lru_evict(pages)
    if policy == "semantic_lru":
        if ram_collection is None or agent_id is None or query_embedding is None:
            raise ValueError(
                "semantic_lru requires ram_collection, agent_id, and query_embedding"
            )
        return semantic_lru_evict(pages, ram_collection, agent_id, query_embedding)

    raise ValueError(f"Unknown replacement policy '{policy}'. Available: {POLICY_NAMES}")
