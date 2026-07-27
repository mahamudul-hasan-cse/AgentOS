"""Shared dataclasses for the memory manager, split out from page_manager.py
so that replacement.py can type-hint against `Page` without a circular import."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class Frame:
    """A unit of *physical* memory: one copy of page content, possibly shared.

    Copy-on-write is built on the classic OS split between a page table (what an
    agent can address) and the frames it points at (what actually occupies
    memory). A frame is referenced by one entry per sharing agent; `refcount`
    is how many page-table entries point here.

    A frame with refcount > 1 is SHARED and immutable in place: writing through
    any of its sharers must allocate a fresh private frame instead (see
    PageManager._copy_on_write), which is what keeps sharers isolated.
    """

    frame_id: str
    content: str
    token_count: int
    embedding: List[float] = field(default_factory=list)
    refcount: int = 0
    sharers: Set[str] = field(default_factory=set)

    @property
    def shared(self) -> bool:
        return self.refcount > 1

    def attach(self, agent_id: str) -> None:
        self.sharers.add(agent_id)
        self.refcount += 1

    def detach(self, agent_id: str) -> int:
        self.sharers.discard(agent_id)
        self.refcount -= 1
        return self.refcount


@dataclass
class Page:
    """One page-table entry: an agent's handle on a frame.

    `content`/`token_count`/`embedding` mirror the frame for convenience (and
    for backwards compatibility with pre-COW callers). The mirror costs nothing:
    Python strings are immutable, so every sharer's `content` is the *same*
    object as the frame's, not a copy.
    """

    page_id: str
    agent_id: str
    content: str
    token_count: int
    embedding: List[float] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    #: frame this entry points at; None only for pages predating COW
    frame_id: Optional[str] = None


@dataclass
class ReadResult:
    page: Page
    page_fault: bool
    evicted_page_id: Optional[str] = None
