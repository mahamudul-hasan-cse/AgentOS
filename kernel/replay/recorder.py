"""Time-travel replay: periodic snapshots of full kernel state.

The StateRecorder observes the syscall dispatcher and captures a snapshot of
the whole system — scheduler queue, per-agent memory (RAM vs swap), provider
allocations, and per-agent quota usage — every N syscalls, plus immediately on
significant events (process termination, quota violation, page eviction).

Snapshots are held in a BOUNDED ring buffer (`collections.deque(maxlen=...)`,
default 200). Once full, appending silently evicts the oldest snapshot, so
memory use stays constant no matter how long the kernel runs. Snapshot ids keep
counting up across evictions, so an id always refers to one specific moment.

Snapshots deliberately store page *identity* (page_id + token_count) rather than
page content: the whole point is a compact timeline, and 200 snapshots holding
full conversation text would defeat the bound.

This module intentionally does NOT import kernel.syscalls — the dispatcher
imports the recorder, so importing back would be circular. Syscall records are
inspected structurally, comparing the string values of their enums.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

DEFAULT_SNAPSHOT_INTERVAL = 5
DEFAULT_MAX_SNAPSHOTS = 200

# syscall/status string values we react to (compared as strings to avoid a
# circular import back into kernel.syscalls)
_TERMINATE_AGENT = "TERMINATE_AGENT"
_MEM_WRITE = "MEM_WRITE"
_MEM_READ = "MEM_READ"
_STATUS_SUCCESS = "success"
_STATUS_QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class Snapshot:
    snapshot_id: int
    timestamp: float
    syscall_id: Optional[str]
    label: str
    processes: List[dict] = field(default_factory=list)
    memory: Dict[str, dict] = field(default_factory=dict)
    resources: Dict[str, dict] = field(default_factory=dict)
    quotas: Dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> Dict[str, Any]:
        """The lightweight form used by the timeline/scrubber UI."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "syscall_id": self.syscall_id,
            "label": self.label,
        }


def _enum_value(value: Any) -> str:
    """Enums here subclass str, but be defensive about plain strings too."""
    return getattr(value, "value", value)


class StateRecorder:
    def __init__(
        self,
        dispatcher: Any,
        interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
    ) -> None:
        if interval < 1:
            raise ValueError("snapshot interval must be >= 1")
        self.dispatcher = dispatcher
        self.interval = interval
        self.max_snapshots = max_snapshots
        # bounded ring buffer: appending past maxlen drops the oldest snapshot
        self.snapshots: Deque[Snapshot] = deque(maxlen=max_snapshots)
        self._next_id = 1
        self._syscalls_seen = 0

    # --- capture ----------------------------------------------------------

    def capture(self, label: str, syscall_id: Optional[str] = None) -> Snapshot:
        """Force a snapshot of current kernel state."""
        snapshot = Snapshot(
            snapshot_id=self._next_id,
            timestamp=time.time(),
            syscall_id=syscall_id,
            label=label,
            processes=self._capture_processes(),
            memory=self._capture_memory(),
            resources=self._capture_resources(),
            quotas=self._capture_quotas(),
        )
        self._next_id += 1
        self.snapshots.append(snapshot)
        return snapshot

    def observe(self, syscall: Any) -> Optional[Snapshot]:
        """Called by the dispatcher after every syscall completes. Captures on a
        significant event, else every `interval` syscalls. Returns the snapshot
        taken, or None."""
        self._syscalls_seen += 1

        event_label = self._significant_event_label(syscall)
        if event_label is not None:
            return self.capture(event_label, syscall_id=getattr(syscall, "syscall_id", None))

        if self._syscalls_seen % self.interval == 0:
            stype = _enum_value(getattr(syscall, "type", "?"))
            agent = getattr(syscall, "agent_id", "?")
            return self.capture(
                f"periodic snapshot after {stype} by {agent}",
                syscall_id=getattr(syscall, "syscall_id", None),
            )
        return None

    @staticmethod
    def _significant_event_label(syscall: Any) -> Optional[str]:
        """A human-readable label if this syscall was a significant event."""
        stype = _enum_value(getattr(syscall, "type", ""))
        status = _enum_value(getattr(syscall, "status", ""))
        agent = getattr(syscall, "agent_id", "?")
        args = getattr(syscall, "args", None) or {}
        result = getattr(syscall, "result", None)
        result = result if isinstance(result, dict) else {}

        # quota violation
        if status == _STATUS_QUOTA_EXCEEDED:
            return f"quota exceeded for {agent}"

        # process termination
        if stype == _TERMINATE_AGENT and status == _STATUS_SUCCESS:
            pid = args.get("pid", "?")
            if result.get("process_found") or result.get("cancelled_llm_call"):
                return f"{pid} terminated"
            return None

        # page eviction (RAM -> ChromaDB swap), from a write or a page fault
        if status == _STATUS_SUCCESS and stype in (_MEM_WRITE, _MEM_READ):
            owner = args.get("target_agent_id") or agent
            evicted = result.get("evicted_page_ids") or []
            single = result.get("evicted_page_id")
            if single:
                evicted = list(evicted) + [single]
            if evicted:
                first = evicted[0]
                extra = f" (+{len(evicted) - 1} more)" if len(evicted) > 1 else ""
                return f"page {first} evicted from {owner}{extra}"
        return None

    # --- state collectors (each defensive: a snapshot must never break a
    # syscall, so collection failures degrade to empty rather than raising) ---

    def _capture_processes(self) -> List[dict]:
        try:
            return [
                {
                    "pid": p.pid,
                    "state": p.state,
                    "arrival_time": p.arrival_time,
                    "estimated_burst": p.estimated_burst,
                    "remaining_burst": p.remaining_burst,
                    "priority": p.priority,
                }
                for p in self.dispatcher.scheduler.queue
            ]
        except Exception:  # noqa: BLE001
            return []

    def _capture_memory(self) -> Dict[str, dict]:
        """Per-agent RAM vs swapped page identity. Reads RAM from the in-memory
        page table and swap from ChromaDB metadata only (no documents), keeping
        snapshots cheap."""
        memory: Dict[str, dict] = {}
        try:
            pm = self.dispatcher.page_manager
            agent_ids = list(pm.ram.keys())
            for agent_id in agent_ids:
                ram_pages = [
                    {"page_id": p.page_id, "token_count": p.token_count}
                    for p in pm.ram[agent_id].values()
                ]
                swapped_pages: List[dict] = []
                try:
                    swapped = pm.swap_collection.get(
                        where={"agent_id": agent_id}, include=["metadatas"]
                    )
                    swapped_pages = [
                        {"page_id": pid, "token_count": (meta or {}).get("token_count")}
                        for pid, meta in zip(swapped["ids"], swapped["metadatas"])
                    ]
                except Exception:  # noqa: BLE001
                    swapped_pages = []
                memory[agent_id] = {
                    "ram_pages": ram_pages,
                    "swapped_pages": swapped_pages,
                    "ram_tokens_used": sum(p["token_count"] for p in ram_pages),
                    "ram_budget_tokens": getattr(pm, "ram_budget_tokens", None),
                }
        except Exception:  # noqa: BLE001
            return memory
        return memory

    def _capture_resources(self) -> Dict[str, dict]:
        try:
            return self.dispatcher.resource_manager.state()
        except Exception:  # noqa: BLE001
            return {}

    def _capture_quotas(self) -> Dict[str, dict]:
        quotas: Dict[str, dict] = {}
        try:
            qm = self.dispatcher.quota_manager
            for agent_id in list(getattr(qm, "_quotas", {}).keys()):
                quotas[agent_id] = qm.usage(agent_id)
        except Exception:  # noqa: BLE001
            return quotas
        return quotas

    # --- retrieval --------------------------------------------------------

    def timeline(self) -> List[dict]:
        return [s.summary() for s in self.snapshots]

    def get(self, snapshot_id: int) -> Optional[Snapshot]:
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None

    # --- diff -------------------------------------------------------------

    def diff(self, id_a: int, id_b: int) -> Dict[str, Any]:
        """Describe what changed between two snapshots. Raises KeyError if
        either id is unknown (evicted from the ring buffer or never taken)."""
        a = self.get(id_a)
        b = self.get(id_b)
        if a is None:
            raise KeyError(f"snapshot {id_a} not found (evicted or never taken)")
        if b is None:
            raise KeyError(f"snapshot {id_b} not found (evicted or never taken)")

        return {
            "from": a.summary(),
            "to": b.summary(),
            "elapsed_seconds": round(b.timestamp - a.timestamp, 3),
            "processes": _diff_processes(a.processes, b.processes),
            "memory": _diff_memory(a.memory, b.memory),
            "resources": _diff_resources(a.resources, b.resources),
            "quotas": _diff_quotas(a.quotas, b.quotas),
        }


def _diff_processes(before: List[dict], after: List[dict]) -> Dict[str, Any]:
    by_pid_a = {p["pid"]: p for p in before}
    by_pid_b = {p["pid"]: p for p in after}

    added = [by_pid_b[pid] for pid in by_pid_b.keys() - by_pid_a.keys()]
    # a process no longer in the queue was terminated/completed
    removed = [by_pid_a[pid] for pid in by_pid_a.keys() - by_pid_b.keys()]
    state_changed = [
        {
            "pid": pid,
            "from_state": by_pid_a[pid]["state"],
            "to_state": by_pid_b[pid]["state"],
        }
        for pid in by_pid_a.keys() & by_pid_b.keys()
        if by_pid_a[pid]["state"] != by_pid_b[pid]["state"]
    ]
    return {
        "added": sorted(added, key=lambda p: p["pid"]),
        "removed": sorted(removed, key=lambda p: p["pid"]),
        "state_changed": sorted(state_changed, key=lambda d: d["pid"]),
    }


def _page_ids(entry: dict, key: str) -> set:
    return {p["page_id"] for p in (entry or {}).get(key, [])}


def _diff_memory(before: Dict[str, dict], after: Dict[str, dict]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for agent_id in set(before) | set(after):
        a = before.get(agent_id, {})
        b = after.get(agent_id, {})
        ram_a, ram_b = _page_ids(a, "ram_pages"), _page_ids(b, "ram_pages")
        swap_a, swap_b = _page_ids(a, "swapped_pages"), _page_ids(b, "swapped_pages")

        entry = {
            # in RAM before, in swap after -> evicted
            "evicted_to_swap": sorted((ram_a - ram_b) & swap_b),
            # in swap before, in RAM after -> paged back in
            "paged_into_ram": sorted((swap_a - swap_b) & ram_b),
            "pages_added": sorted((ram_b | swap_b) - (ram_a | swap_a)),
            "pages_removed": sorted((ram_a | swap_a) - (ram_b | swap_b)),
            "ram_tokens_delta": (b.get("ram_tokens_used") or 0)
            - (a.get("ram_tokens_used") or 0),
        }
        if any(entry[k] for k in ("evicted_to_swap", "paged_into_ram", "pages_added", "pages_removed")) or entry["ram_tokens_delta"]:
            changes[agent_id] = entry
    return changes


def _diff_resources(before: Dict[str, dict], after: Dict[str, dict]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for provider in set(before) | set(after):
        a = before.get(provider, {})
        b = after.get(provider, {})
        delta = (b.get("allocated") or 0) - (a.get("allocated") or 0)
        if delta or a.get("safe") != b.get("safe"):
            changes[provider] = {
                "allocated_delta": delta,
                "allocated_before": a.get("allocated"),
                "allocated_after": b.get("allocated"),
                "safe_before": a.get("safe"),
                "safe_after": b.get("safe"),
            }
    return changes


def _diff_quotas(before: Dict[str, dict], after: Dict[str, dict]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for agent_id in set(before) | set(after):
        a = before.get(agent_id, {})
        b = after.get(agent_id, {})
        pages_delta = (b.get("pages_used") or 0) - (a.get("pages_used") or 0)
        calls_delta = (b.get("calls_in_window") or 0) - (a.get("calls_in_window") or 0)
        if pages_delta or calls_delta:
            changes[agent_id] = {
                "pages_used_delta": pages_delta,
                "calls_in_window_delta": calls_delta,
                "pages_used_after": b.get("pages_used"),
                "calls_in_window_after": b.get("calls_in_window"),
            }
    return changes
