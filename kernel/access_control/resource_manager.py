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
from typing import Any, Dict, List, Optional

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
        # agents whose most recent request for this pool was refused and are
        # therefore blocked on it. This is what turns allocation state into a
        # wait-for graph: a waiter has an edge to every current holder.
        self.waiting: Dict[str, int] = {}

    def allocated(self) -> int:
        return sum(self.allocation.values())

    def available(self) -> int:
        return self.total - self.allocated()

    def holders(self) -> List[str]:
        return [agent for agent, units in self.allocation.items() if units > 0]


class ResourceManager:
    def __init__(
        self,
        capacities: Optional[Dict[str, int]] = None,
        default_capacity: int = DEFAULT_CAPACITY,
        avoidance_enabled: bool = True,
    ) -> None:
        self._default_capacity = default_capacity
        source = capacities if capacities is not None else DEFAULT_CAPACITIES
        self._pools: Dict[str, ProviderPool] = {
            name: ProviderPool(cap) for name, cap in source.items()
        }
        self._lock = asyncio.Lock()
        # Deadlock AVOIDANCE (Banker's Algorithm). While enabled, a grant that
        # would leave the pool in an unsafe state is refused, so a true deadlock
        # can essentially never form — which also means the deadlock DETECTOR
        # would have nothing to find. Turning this off makes the manager hand
        # out slots greedily (still bounded by physical capacity), letting real
        # circular waits develop so detection and recovery can be exercised.
        self.avoidance_enabled = avoidance_enabled

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
                # blocked on this pool: it is held by whoever currently owns it
                pool.waiting[agent_id] = units
                return False

            claim = max_claim if max_claim is not None else 1
            declared = max(pool.max_claim.get(agent_id, 0), claim, new_allocation)

            prev_claim = pool.max_claim.get(agent_id)
            pool.allocation[agent_id] = new_allocation
            pool.max_claim[agent_id] = declared

            # With avoidance disabled the safety check is skipped entirely and
            # the grant goes through greedily.
            if not self.avoidance_enabled or self._is_safe(pool):
                pool.peak_allocated = max(pool.peak_allocated, pool.allocated())
                pool.waiting.pop(agent_id, None)  # no longer blocked
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
            pool.waiting[agent_id] = units
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
            pool.waiting.pop(agent_id, None)

    def release_all(self, agent_id: str) -> Dict[str, int]:
        """Drop every allocation and pending wait held by an agent, across all
        pools. Used by deadlock recovery when a victim is terminated: its
        resources must go back to the pool for the cycle to actually break."""
        freed: Dict[str, int] = {}
        for name, pool in self._pools.items():
            units = pool.allocation.pop(agent_id, None)
            pool.max_claim.pop(agent_id, None)
            pool.waiting.pop(agent_id, None)
            if units:
                freed[name] = units
        return freed

    def clear_wait(self, agent_id: str, provider: Optional[str] = None) -> None:
        """Forget that an agent is blocked (it gave up, or was served)."""
        pools = [self._pool(provider)] if provider else list(self._pools.values())
        for pool in pools:
            pool.waiting.pop(agent_id, None)

    def set_avoidance(self, enabled: bool) -> bool:
        self.avoidance_enabled = enabled
        return self.avoidance_enabled

    def waiting_state(self) -> Dict[str, Dict[str, int]]:
        """provider -> {agent: units it is blocked waiting for}."""
        return {
            name: dict(pool.waiting)
            for name, pool in self._pools.items()
            if pool.waiting
        }

    def holdings(self) -> Dict[str, Dict[str, int]]:
        """agent -> {provider: units held}, the inverse of the allocation view."""
        out: Dict[str, Dict[str, int]] = {}
        for name, pool in self._pools.items():
            for agent, units in pool.allocation.items():
                if units > 0:
                    out.setdefault(agent, {})[name] = units
        return out

    def state(self) -> Dict[str, Any]:
        """Snapshot of allocation/availability/safe-state per provider."""
        return {
            name: {
                "waiting": dict(pool.waiting),
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
