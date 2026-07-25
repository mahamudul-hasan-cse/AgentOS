"""Manages a queue of agent processes and dispatches them to a chosen
scheduling algorithm."""

from __future__ import annotations

from typing import List, Optional, Sequence

from .algorithms import (
    DEFAULT_MLFQ_QUANTUMS,
    Process,
    TimeSlice,
    fcfs,
    mlfq,
    priority_scheduling,
    round_robin,
)

DEFAULT_QUANTUM = 4.0
ALGORITHM_NAMES = ("fcfs", "round_robin", "priority", "mlfq")


class UnknownAlgorithmError(ValueError):
    """Raised when an unrecognized algorithm name is requested."""


class Scheduler:
    def __init__(self, processes: Optional[Sequence[Process]] = None):
        self.queue: List[Process] = list(processes) if processes else []

    def add_process(self, process: Process) -> None:
        self.queue.append(process)

    def terminate(self, pid: str) -> bool:
        """SIGKILL-style termination: immediately mark any process with this pid
        as 'terminated' and drop it from the ready/waiting queue. Returns True if
        a matching process was found. The Process object itself survives (a caller
        holding a reference sees state == 'terminated'); it is just no longer
        schedulable."""
        found = False
        for process in self.queue:
            if process.pid == pid:
                process.state = "terminated"
                process.remaining_burst = 0.0
                found = True
        self.queue = [p for p in self.queue if p.pid != pid]
        return found

    def run(
        self,
        algorithm: str,
        quantum: float = DEFAULT_QUANTUM,
        mlfq_quantums: Optional[Sequence[float]] = None,
    ) -> List[TimeSlice]:
        if algorithm == "fcfs":
            return fcfs(self.queue)
        if algorithm == "round_robin":
            return round_robin(self.queue, quantum=quantum)
        if algorithm == "priority":
            return priority_scheduling(self.queue)
        if algorithm == "mlfq":
            return mlfq(self.queue, quantums=mlfq_quantums or DEFAULT_MLFQ_QUANTUMS)
        raise UnknownAlgorithmError(
            f"Unknown algorithm '{algorithm}'. Available: {', '.join(ALGORITHM_NAMES)}"
        )
