"""Per-agent quotas — a second dimension on top of the per-provider resource
pools (Banker's Algorithm) from Phase 6.

Where the ResourceManager limits *concurrent* usage of a shared provider, the
QuotaManager limits what a *single agent* may consume over time:

- max memory pages: distinct page ids the agent owns in the PageManager.
- max LLM calls per rolling 60-second window.

Rate limiting uses a SLIDING-WINDOW LOG: we keep the timestamps of an agent's
recent calls and count how many fall within the last `window_seconds`. Chosen
over a token bucket because it maps directly onto the "calls per rolling
60-second window" spec and needs no separate refill bookkeeping. `time.monotonic`
is used so the window is immune to wall-clock adjustments; callers may pass an
explicit `now` for deterministic testing.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set

DEFAULT_MAX_PAGES = 20
DEFAULT_MAX_CALLS_PER_MINUTE = 10
WINDOW_SECONDS = 60.0


class QuotaExceeded(Exception):
    """Raised when an operation would exceed an agent's quota. The dispatcher
    traps this into a QUOTA_EXCEEDED syscall status rather than crashing."""


@dataclass
class AgentQuota:
    max_pages: int = DEFAULT_MAX_PAGES
    max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE
    page_ids: Set[str] = field(default_factory=set)
    call_times: Deque[float] = field(default_factory=deque)


class QuotaManager:
    def __init__(
        self,
        default_max_pages: int = DEFAULT_MAX_PAGES,
        default_max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        self.default_max_pages = default_max_pages
        self.default_max_calls_per_minute = default_max_calls_per_minute
        self.window_seconds = window_seconds
        self._quotas: Dict[str, AgentQuota] = {}

    def _quota(self, agent_id: str) -> AgentQuota:
        quota = self._quotas.get(agent_id)
        if quota is None:
            quota = AgentQuota(
                max_pages=self.default_max_pages,
                max_calls_per_minute=self.default_max_calls_per_minute,
            )
            self._quotas[agent_id] = quota
        return quota

    # --- memory-page quota ------------------------------------------------

    def can_write_page(self, agent_id: str, page_id: str) -> bool:
        """True if the agent may (over)write this page without exceeding its
        page quota. Overwriting an already-owned page never counts as new."""
        quota = self._quota(agent_id)
        if page_id in quota.page_ids:
            return True
        return len(quota.page_ids) < quota.max_pages

    def record_page(self, agent_id: str, page_id: str) -> None:
        self._quota(agent_id).page_ids.add(page_id)

    # --- call-rate quota (sliding window) ---------------------------------

    def _purge_window(self, quota: AgentQuota, now: float) -> None:
        cutoff = now - self.window_seconds
        while quota.call_times and quota.call_times[0] <= cutoff:
            quota.call_times.popleft()

    def try_consume_call(self, agent_id: str, now: Optional[float] = None) -> bool:
        """Record one LLM call if the agent is under its per-window limit.
        Returns False (and records nothing) if the limit is already reached.
        Synchronous and await-free, so it runs atomically within one event-loop
        step — no lock needed under asyncio."""
        now = time.monotonic() if now is None else now
        quota = self._quota(agent_id)
        self._purge_window(quota, now)
        if len(quota.call_times) >= quota.max_calls_per_minute:
            return False
        quota.call_times.append(now)
        return True

    # --- administration ---------------------------------------------------

    def set_quota(
        self,
        agent_id: str,
        max_pages: Optional[int] = None,
        max_calls_per_minute: Optional[int] = None,
    ) -> AgentQuota:
        """Adjust an agent's limits. Low-level setter — callers gate access
        (the dispatcher exposes this only through the KERNEL-only SET_QUOTA
        syscall, the same way other privileged operations are gated)."""
        quota = self._quota(agent_id)
        if max_pages is not None:
            quota.max_pages = max_pages
        if max_calls_per_minute is not None:
            quota.max_calls_per_minute = max_calls_per_minute
        return quota

    def usage(self, agent_id: str, now: Optional[float] = None) -> Dict[str, object]:
        """Current usage vs. limit for both dimensions."""
        now = time.monotonic() if now is None else now
        quota = self._quota(agent_id)
        self._purge_window(quota, now)
        return {
            "agent_id": agent_id,
            "pages_used": len(quota.page_ids),
            "max_pages": quota.max_pages,
            "calls_in_window": len(quota.call_times),
            "max_calls_per_minute": quota.max_calls_per_minute,
            "window_seconds": self.window_seconds,
        }
