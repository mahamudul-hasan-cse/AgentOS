"""Provider rate-limit pools with deadlock avoidance (Banker's Algorithm).

Each LLM provider is modelled as a finite resource: a pool with `total`
concurrent-request slots. Before an LLM_CALL is dispatched, the agent requests
a slot and declares a max claim on that provider (default 1). A request is
granted only if the resulting allocation leaves the system in a *safe* state —
one where every currently-allocated agent could still reach its declared max
claim in some serial order — otherwise it is refused and the caller queues,
denies, or falls back to another provider.

This is the single-resource-type form of the Banker's Algorithm, applied
independently per provider.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from kernel.drivers import DriverError

# Default per-provider concurrent-request capacities. Free-tier limits vary and
# change over time — treat these as tunable placeholders.
DEFAULT_CAPACITIES: Dict[str, int] = {
    "groq": 30,
    "deepseek": 30,
    "gemini": 15,
    "ollama": 4,
}
DEFAULT_CAPACITY = 30


class ResourceUnavailable(DriverError):
    """Raised/flagged when a provider cannot grant a slot without becoming
    unsafe or exceeding capacity. Subclasses DriverError so it flows through the
    dispatcher's LLM_CALL fallback path like any other provider failure."""


class ProviderPool:
    def __init__(self, total: int) -> None:
        self.total = total
        self.allocation: Dict[str, int] = {}
        self.max_claim: Dict[str, int] = {}
        self.peak_allocated = 0

    def allocated(self) -> int:
        return sum(self.allocation.values())

    def available(self) -> int:
        return self.total - self.allocated()


class ResourceManager:
    def __init__(
        self,
        capacities: Optional[Dict[str, int]] = None,
        default_capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        self._default_capacity = default_capacity
        source = capacities if capacities is not None else DEFAULT_CAPACITIES
        self._pools: Dict[str, ProviderPool] = {
            name: ProviderPool(cap) for name, cap in source.items()
        }
        self._lock = asyncio.Lock()

    def _pool(self, provider: str) -> ProviderPool:
        pool = self._pools.get(provider)
        if pool is None:
            pool = ProviderPool(self._default_capacity)
            self._pools[provider] = pool
        return pool

    @staticmethod
    def _is_safe(pool: ProviderPool) -> bool:
        """Single-resource safety check: is there an order in which every
        allocated agent can reach its max claim and release?"""
        work = pool.available()
        allocation = dict(pool.allocation)
        need = {a: pool.max_claim.get(a, 0) - allocation[a] for a in allocation}
        finished = {a: False for a in allocation}

        progressed = True
        while progressed:
            progressed = False
            for agent_id in allocation:
                if not finished[agent_id] and need[agent_id] <= work:
                    work += allocation[agent_id]
                    finished[agent_id] = True
                    progressed = True
        return all(finished.values())

    async def request(
        self,
        agent_id: str,
        provider: str,
        units: int = 1,
        max_claim: Optional[int] = None,
    ) -> bool:
        """Try to grant `units` slots on `provider` to `agent_id`. Returns True
        if granted (state remains safe), False if it would exceed capacity or
        enter an unsafe state."""
        async with self._lock:
            pool = self._pool(provider)
            current = pool.allocation.get(agent_id, 0)
            new_allocation = current + units

            # can't hand out more than physically exists right now
            if units > pool.available():
                return False

            claim = max_claim if max_claim is not None else 1
            declared = max(pool.max_claim.get(agent_id, 0), claim, new_allocation)

            prev_claim = pool.max_claim.get(agent_id)
            pool.allocation[agent_id] = new_allocation
            pool.max_claim[agent_id] = declared

            if self._is_safe(pool):
                pool.peak_allocated = max(pool.peak_allocated, pool.allocated())
                return True

            # unsafe — roll back the tentative allocation
            if current == 0:
                pool.allocation.pop(agent_id, None)
            else:
                pool.allocation[agent_id] = current
            if prev_claim is None:
                pool.max_claim.pop(agent_id, None)
            else:
                pool.max_claim[agent_id] = prev_claim
            return False

    async def release(self, agent_id: str, provider: str, units: int = 1) -> None:
        async with self._lock:
            pool = self._pool(provider)
            current = pool.allocation.get(agent_id, 0)
            remaining = current - units
            if remaining > 0:
                pool.allocation[agent_id] = remaining
            else:
                pool.allocation.pop(agent_id, None)
                pool.max_claim.pop(agent_id, None)

    def state(self) -> Dict[str, Any]:
        """Snapshot of allocation/availability/safe-state per provider."""
        return {
            name: {
                "total": pool.total,
                "allocated": pool.allocated(),
                "available": pool.available(),
                "peak_allocated": pool.peak_allocated,
                "safe": self._is_safe(pool),
                "allocation": dict(pool.allocation),
                "max_claim": dict(pool.max_claim),
            }
            for name, pool in self._pools.items()
        }
