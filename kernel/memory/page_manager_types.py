"""Shared dataclasses for the memory manager, split out from page_manager.py
so that replacement.py can type-hint against `Page` without a circular import."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Page:
    page_id: str
    agent_id: str
    content: str
    token_count: int
    embedding: List[float] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)


@dataclass
class ReadResult:
    page: Page
    page_fault: bool
    evicted_page_id: Optional[str] = None
