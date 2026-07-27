"""Scheduling algorithms operating over a queue of agent "process" objects.

Each algorithm is a pure function: it consumes a list of `Process` objects
(mutating their `state`/`remaining_burst` as they run) and returns the
resulting execution timeline as a list of `TimeSlice`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

DEFAULT_MLFQ_QUANTUMS = (4.0, 8.0, 16.0)


@dataclass
class Process:
    """A scheduled agent request.

    `priority` follows the OS convention where a lower number means higher
    priority (0 = highest).
    """

    pid: str
    arrival_time: float
    estimated_burst: float
    priority: int = 0
    state: str = "waiting"  # waiting | ready | running | zombie | terminated
    # Parent in the process hierarchy. None means no parent (a root process);
    # every other process descends from the "init" process. A terminated child
    # lingers as a `zombie` holding `exit_status` until its parent reaps it.
    parent_pid: Optional[str] = None
    exit_status: Optional[int] = None
    remaining_burst: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_burst = self.estimated_burst


@dataclass
class TimeSlice:
    pid: str
    start: float
    end: float


def _sorted_by_arrival(processes: Sequence[Process]) -> List[Process]:
    return sorted(processes, key=lambda p: (p.arrival_time, p.pid))


def fcfs(processes: Sequence[Process]) -> List[TimeSlice]:
    """First-Come, First-Served: non-preemptive, runs each process to completion
    in arrival order."""
    timeline: List[TimeSlice] = []
    current_time = 0.0
    for process in _sorted_by_arrival(processes):
        process.state = "running"
        start = max(current_time, process.arrival_time)
        end = start + process.remaining_burst
        timeline.append(TimeSlice(pid=process.pid, start=start, end=end))
        process.remaining_burst = 0.0
        process.state = "terminated"
        current_time = end
    return timeline


def round_robin(processes: Sequence[Process], quantum: float) -> List[TimeSlice]:
    """Round Robin with a token-based quantum: each process runs for at most
    `quantum` tokens before being preempted and sent to the back of the queue."""
    if quantum <= 0:
        raise ValueError("quantum must be positive")

    timeline: List[TimeSlice] = []
    pending = _sorted_by_arrival(processes)
    if not pending:
        return timeline

    ready_queue: List[Process] = []
    idx = 0
    current_time = pending[0].arrival_time

    def admit_arrivals(up_to_time: float) -> None:
        nonlocal idx
        while idx < len(pending) and pending[idx].arrival_time <= up_to_time:
            pending[idx].state = "ready"
            ready_queue.append(pending[idx])
            idx += 1

    admit_arrivals(current_time)

    while ready_queue or idx < len(pending):
        if not ready_queue:
            current_time = pending[idx].arrival_time
            admit_arrivals(current_time)
            continue

        process = ready_queue.pop(0)
        process.state = "running"
        run_time = min(quantum, process.remaining_burst)
        start = current_time
        end = start + run_time
        timeline.append(TimeSlice(pid=process.pid, start=start, end=end))
        process.remaining_burst -= run_time
        current_time = end

        admit_arrivals(current_time)

        if process.remaining_burst > 0:
            process.state = "ready"
            ready_queue.append(process)
        else:
            process.state = "terminated"

    return timeline


def priority_scheduling(processes: Sequence[Process]) -> List[TimeSlice]:
    """Non-preemptive priority scheduling. Lower `priority` value runs first;
    ties broken by arrival time then pid."""
    timeline: List[TimeSlice] = []
    remaining = _sorted_by_arrival(processes)
    if not remaining:
        return timeline

    current_time = remaining[0].arrival_time

    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available:
            current_time = min(p.arrival_time for p in remaining)
            continue

        process = min(available, key=lambda p: (p.priority, p.arrival_time, p.pid))
        process.state = "running"
        start = max(current_time, process.arrival_time)
        end = start + process.remaining_burst
        timeline.append(TimeSlice(pid=process.pid, start=start, end=end))
        process.remaining_burst = 0.0
        process.state = "terminated"
        current_time = end
        remaining.remove(process)

    return timeline


def mlfq(
    processes: Sequence[Process], quantums: Optional[Sequence[float]] = None
) -> List[TimeSlice]:
    """Multi-Level Feedback Queue: new processes enter at level 0 (highest
    priority, smallest quantum). A process that doesn't finish within its
    level's quantum is demoted one level. The lowest level runs to completion
    (no further demotion possible). Higher levels always preempt lower ones."""
    levels_quantums = list(quantums) if quantums else list(DEFAULT_MLFQ_QUANTUMS)
    num_levels = len(levels_quantums)
    if num_levels == 0:
        raise ValueError("mlfq requires at least one queue level")

    timeline: List[TimeSlice] = []
    pending = _sorted_by_arrival(processes)
    if not pending:
        return timeline

    levels: List[List[Process]] = [[] for _ in range(num_levels)]
    idx = 0
    current_time = pending[0].arrival_time

    def admit_arrivals(up_to_time: float) -> None:
        nonlocal idx
        while idx < len(pending) and pending[idx].arrival_time <= up_to_time:
            pending[idx].state = "ready"
            levels[0].append(pending[idx])
            idx += 1

    admit_arrivals(current_time)

    while any(levels) or idx < len(pending):
        level = next((lvl for lvl in range(num_levels) if levels[lvl]), None)
        if level is None:
            current_time = pending[idx].arrival_time
            admit_arrivals(current_time)
            continue

        process = levels[level].pop(0)
        process.state = "running"
        is_lowest_level = level == num_levels - 1
        run_time = (
            process.remaining_burst
            if is_lowest_level
            else min(levels_quantums[level], process.remaining_burst)
        )
        start = current_time
        end = start + run_time
        timeline.append(TimeSlice(pid=process.pid, start=start, end=end))
        process.remaining_burst -= run_time
        current_time = end

        admit_arrivals(current_time)

        if process.remaining_burst > 0:
            process.state = "ready"
            next_level = min(level + 1, num_levels - 1)
            levels[next_level].append(process)
        else:
            process.state = "terminated"

    return timeline
