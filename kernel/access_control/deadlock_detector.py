"""Deadlock DETECTION and RECOVERY — the complement to the Banker's Algorithm.

Section 3.7 of the plan implements deadlock *avoidance*: the ResourceManager
refuses any grant that would leave a pool in an unsafe state, so a circular wait
essentially cannot form. Avoidance is conservative — it denies requests that
might have been fine — so the classic alternative is to allow deadlocks and then
detect and recover from them. This module is that alternative.

**A note on testability.** While `ResourceManager.avoidance_enabled` is True (the
default) this detector should, correctly, never find anything: avoidance is
doing its job upstream. That is not a bug, it is the point — the two strategies
are alternatives, not layers. Set `avoidance_enabled = False` (POST
/resources/mode) to let the manager grant greedily so real circular waits can
form and be detected.

Wait-for graph
--------------
A node is an agent. An edge A -> B means "A is blocked on a resource that B
currently holds". Edges are derived from the ResourceManager: for every pool
where A is recorded as waiting, an edge is drawn from A to each current holder
of that pool. A cycle in this graph is a deadlock: every agent in it is waiting
for something only another cycle member can release.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

DEFAULT_DETECTION_INTERVAL = 5.0
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def configured_interval(default: float = DEFAULT_DETECTION_INTERVAL) -> float:
    """Read `deadlock.interval_seconds` from kernel/config.yaml.

    Kept configurable because the default 5s is good for a running system but
    awkward for a demo: the monitor recovers a deadlock within one interval, so
    a cycle can vanish before anyone sees it on the dashboard. Raising this to
    30-60s makes the deadlocked state comfortably observable.
    """
    try:
        import yaml

        if not CONFIG_PATH.exists():
            return default
        with open(CONFIG_PATH, "r") as handle:
            config = yaml.safe_load(handle) or {}
        section = config.get("deadlock") or {}
        value = section.get("interval_seconds")
        return float(value) if value is not None and float(value) > 0 else default
    except Exception:  # noqa: BLE001 — a broken config must never break startup
        return default


@dataclass
class WaitForGraph:
    #: agent -> set of agents it is waiting on
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    #: agent -> {provider: units held}
    holdings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: provider -> {agent: units awaited}
    waiting: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def nodes(self) -> List[str]:
        found: Set[str] = set(self.edges) | set(self.holdings)
        for waiters in self.waiting.values():
            found |= set(waiters)
        for targets in self.edges.values():
            found |= targets
        return sorted(found)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "agent_id": agent,
                    "holds": self.holdings.get(agent, {}),
                    "waiting_on": sorted(self.edges.get(agent, set())),
                }
                for agent in self.nodes()
            ],
            "edges": sorted(
                ({"from": src, "to": dst} for src, dsts in self.edges.items() for dst in dsts),
                key=lambda e: (e["from"], e["to"]),
            ),
            "waiting": {p: dict(w) for p, w in self.waiting.items()},
        }


def find_cycle(edges: Dict[str, Set[str]]) -> Optional[List[str]]:
    """Depth-first search returning the actual members of a cycle (in order),
    or None. Iterative colouring: WHITE unvisited, GREY on the current stack,
    BLACK finished. Meeting a GREY node closes a cycle, and the path from that
    node to the current one *is* the cycle."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {}
    stack_path: List[str] = []

    def visit(node: str) -> Optional[List[str]]:
        colour[node] = GREY
        stack_path.append(node)
        for neighbour in sorted(edges.get(node, ())):
            state = colour.get(neighbour, WHITE)
            if state == GREY:
                # found it: slice the current path from the repeated node
                start = stack_path.index(neighbour)
                return stack_path[start:]
            if state == WHITE:
                found = visit(neighbour)
                if found is not None:
                    return found
        stack_path.pop()
        colour[node] = BLACK
        return None

    for node in sorted(edges):
        if colour.get(node, WHITE) == WHITE:
            cycle = visit(node)
            if cycle is not None:
                return cycle
    return None


@dataclass
class DetectionResult:
    timestamp: float
    deadlocked: bool
    cycle: List[str]
    graph: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "deadlocked": self.deadlocked,
            "cycle": self.cycle,
            "graph": self.graph,
        }


class DeadlockDetector:
    def __init__(
        self,
        resource_manager: Any,
        scheduler: Any = None,
        terminate: Optional[Callable[[str], Any]] = None,
        interval: Optional[float] = None,
    ) -> None:
        # None => take it from kernel/config.yaml (deadlock.interval_seconds)
        if interval is None:
            interval = configured_interval()
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        #: async callable(pid) used to kill a victim; wired to the dispatcher's
        #: TERMINATE_AGENT path so recovery goes through the normal syscall route
        self._terminate = terminate
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self.detection_runs = 0
        self.recoveries: List[Dict[str, Any]] = []

    # --- graph + detection -------------------------------------------------

    def build_graph(self) -> WaitForGraph:
        waiting = self.resource_manager.waiting_state()
        holdings = self.resource_manager.holdings()

        edges: Dict[str, Set[str]] = {}
        for provider, waiters in waiting.items():
            pool_holders = self.resource_manager._pool(provider).holders()
            for waiter in waiters:
                for holder in pool_holders:
                    if holder == waiter:
                        continue  # never an edge to yourself
                    edges.setdefault(waiter, set()).add(holder)
        return WaitForGraph(edges=edges, holdings=holdings, waiting=waiting)

    def detect(self) -> DetectionResult:
        """One on-demand detection pass."""
        self.detection_runs += 1
        graph = self.build_graph()
        cycle = find_cycle(graph.edges) or []
        return DetectionResult(
            timestamp=time.time(),
            deadlocked=bool(cycle),
            cycle=cycle,
            graph=graph.as_dict(),
        )

    def status(self) -> Dict[str, Any]:
        result = self.detect()
        return {
            "deadlocked": result.deadlocked,
            "cycle": result.cycle,
            "detection_runs": self.detection_runs,
            "avoidance_enabled": self.resource_manager.avoidance_enabled,
            "interval_seconds": self.interval,
            "monitoring": self._task is not None and not self._task.done(),
            "recoveries": len(self.recoveries),
        }

    # --- victim selection + recovery ---------------------------------------

    def select_victim(self, cycle: List[str]) -> Optional[str]:
        """Pick which member of the cycle to sacrifice.

        Policy, in order:
          1. fewest resources held  — killing it frees the least, so we destroy
             the least work while still breaking the cycle;
          2. lowest scheduling priority — in this codebase a HIGHER `priority`
             number means lower priority, so we prefer the largest value;
          3. most recent arrival — the newest process has done the least work,
             so rolling it back costs least.
        Ties beyond that fall back to pid order so selection stays deterministic.

        Real operating systems use the same shape of cost-based heuristic when
        choosing a victim for termination or rollback (Silberschatz lists
        process priority, elapsed compute time, resources held and resources
        needed among the usual factors); there is no single correct answer, only
        a cheapest-to-abort estimate.
        """
        if not cycle:
            return None
        holdings = self.resource_manager.holdings()

        def cost(agent: str):
            held = sum(holdings.get(agent, {}).values())
            process = self.scheduler.get(agent) if self.scheduler else None
            # priority: larger number == lower priority == better victim
            priority = process.priority if process is not None else 0
            arrival = process.arrival_time if process is not None else 0.0
            return (held, -priority, -arrival, agent)

        return min(cycle, key=cost)

    async def recover(self, cycle: Optional[List[str]] = None) -> Dict[str, Any]:
        """Break a detected cycle by terminating one member, then re-detect to
        confirm the cycle is actually gone."""
        if cycle is None:
            cycle = self.detect().cycle
        if not cycle:
            return {"recovered": False, "reason": "no deadlock", "cycle": []}

        victim = self.select_victim(cycle)
        if victim is None:
            return {"recovered": False, "reason": "no victim", "cycle": cycle}

        holdings_before = dict(self.resource_manager.holdings().get(victim, {}))
        if self._terminate is not None:
            await self._terminate(victim)
        # Release whatever the victim still held: terminating a process must
        # return its resources, otherwise the cycle would survive its death.
        freed = self.resource_manager.release_all(victim)

        after = self.detect()
        record = {
            "recovered": not after.deadlocked,
            "victim": victim,
            "cycle": cycle,
            "held_by_victim": holdings_before,
            "freed": freed,
            "remaining_cycle": after.cycle,
            "timestamp": time.time(),
        }
        self.recoveries.append(record)
        return record

    # --- background monitoring ---------------------------------------------

    async def _monitor(self, on_event: Optional[Callable] = None) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                if on_event is not None:
                    await on_event()
                else:
                    result = self.detect()
                    if result.deadlocked:
                        await self.recover(result.cycle)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — monitoring must never die
                pass

    def start(self, on_event: Optional[Callable] = None) -> None:
        """Begin periodic detection in the background (no-op if already running)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.ensure_future(self._monitor(on_event))

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except BaseException:  # noqa: BLE001 — cancellation expected
            pass
        self._task = None
