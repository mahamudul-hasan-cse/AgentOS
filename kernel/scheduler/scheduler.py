"""Manages a queue of agent processes and dispatches them to a chosen
scheduling algorithm."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .algorithms import (
    DEFAULT_MLFQ_QUANTUMS,
    Process,
    TimeSlice,
    fcfs,
    mlfq,
    mlfq_boost,
    priority_aging,
    priority_scheduling,
    round_robin,
)

DEFAULT_QUANTUM = 4.0
ALGORITHM_NAMES = (
    "fcfs",
    "round_robin",
    "priority",
    "priority_aging",
    "mlfq",
    "mlfq_boost",
)

#: The root of the process hierarchy, ancestor of everything (PID 1 in a real OS).
INIT_PID = "init"
ZOMBIE = "zombie"


class UnknownAlgorithmError(ValueError):
    """Raised when an unrecognized algorithm name is requested."""


class Scheduler:
    def __init__(self, processes: Optional[Sequence[Process]] = None):
        self.queue: List[Process] = list(processes) if processes else []

    def add_process(self, process: Process) -> None:
        self.queue.append(process)

    # --- process hierarchy -------------------------------------------------

    def ensure_init(self) -> Process:
        """Return the init process, creating it if absent. init is the ancestor
        of every process and is itself parentless."""
        existing = self.get(INIT_PID)
        if existing is not None:
            return existing
        init = Process(
            pid=INIT_PID, arrival_time=0.0, estimated_burst=0.0, state="ready"
        )
        self.queue.insert(0, init)
        return init

    def get(self, pid: str) -> Optional[Process]:
        for process in self.queue:
            if process.pid == pid:
                return process
        return None

    def children_of(self, pid: str) -> List[Process]:
        """Direct children.

        init additionally adopts, for display purposes, any process that would
        otherwise be unreachable from the root: one with no parent at all, or
        one whose parent_pid names something that is not (or is no longer) a
        process. Without this a subtree could vanish from the tree while still
        appearing in the process table, which is exactly the inconsistency the
        hierarchy is meant to prevent."""
        children = [p for p in self.queue if p.parent_pid == pid]
        if pid == INIT_PID:
            known = {p.pid for p in self.queue}
            children += [
                p
                for p in self.queue
                if p.pid != INIT_PID
                and (p.parent_pid is None or p.parent_pid not in known)
            ]
        return sorted(children, key=lambda p: p.pid)

    def descendants(self, pid: str) -> List[Process]:
        out: List[Process] = []
        for child in self.children_of(pid):
            out.append(child)
            out.extend(self.descendants(child.pid))
        return out

    def spawn(
        self,
        pid: str,
        parent_pid: str,
        estimated_burst: float = 0.0,
        priority: int = 0,
        arrival_time: float = 0.0,
    ) -> Process:
        if self.get(pid) is not None:
            raise ValueError(f"process '{pid}' already exists")
        self.ensure_init()
        # An agent that forks is itself a process. Callers routinely spawn from
        # a bare agent id that was never registered (e.g. "root"), so
        # materialise the parent under init rather than leaving its children
        # dangling off a pid that does not exist.
        if parent_pid != INIT_PID and self.get(parent_pid) is None:
            self.queue.append(
                Process(
                    pid=parent_pid,
                    arrival_time=arrival_time,
                    estimated_burst=0.0,
                    state="ready",
                    parent_pid=INIT_PID,
                )
            )
        child = Process(
            pid=pid,
            arrival_time=arrival_time,
            estimated_burst=estimated_burst,
            priority=priority,
            state="ready",
            parent_pid=parent_pid,
        )
        self.queue.append(child)
        return child

    def get_tree(self, root_pid: str = INIT_PID) -> Dict[str, Any]:
        """The hierarchy as nested JSON-serializable dicts, rooted at init."""
        self.ensure_init()

        def node(process: Process) -> Dict[str, Any]:
            return {
                "pid": process.pid,
                "state": process.state,
                "parent_pid": process.parent_pid,
                "priority": process.priority,
                "remaining_burst": process.remaining_burst,
                "exit_status": process.exit_status,
                "children": [node(c) for c in self.children_of(process.pid)],
            }

        root = self.get(root_pid)
        return node(root) if root is not None else {}

    # --- lifecycle ---------------------------------------------------------

    def _remove(self, pid: str) -> None:
        self.queue = [p for p in self.queue if p.pid != pid]

    def terminate(self, pid: str, exit_status: int = 0) -> Dict[str, Any]:
        """SIGKILL-style termination.

        Children are NOT killed with their parent. They are reparented to init,
        matching real OS behaviour where orphans are adopted by PID 1 rather
        than destroyed; cascading termination is opt-in via `kill_tree`.

        The dying process itself becomes a `zombie` — retaining its exit status
        so the parent can still read it — unless nobody is left to reap it. A
        process with no parent, or whose parent is init, is removed immediately,
        because init reaps its children automatically (as PID 1 does).
        """
        process = self.get(pid)
        if process is None:
            return {"found": False, "zombie": False, "reparented": [], "reaped": []}

        reparented: List[str] = []
        reaped: List[str] = []
        for child in self.children_of(pid):
            if child.pid == pid:
                continue
            child.parent_pid = INIT_PID
            reparented.append(child.pid)
            # a zombie handed to init is reaped by it at once
            if child.state == ZOMBIE:
                self._remove(child.pid)
                reaped.append(child.pid)

        process.remaining_burst = 0.0
        process.exit_status = exit_status

        parent_pid = process.parent_pid
        auto_reaped = parent_pid is None or parent_pid == INIT_PID or pid == INIT_PID
        if auto_reaped:
            process.state = "terminated"
            self._remove(pid)
        else:
            process.state = ZOMBIE

        return {
            "found": True,
            "zombie": not auto_reaped,
            "reparented": reparented,
            "reaped": reaped,
            "exit_status": exit_status,
        }

    def reap(self, parent_pid: str, child_pid: Optional[str] = None) -> Optional[Process]:
        """wait(): remove one of `parent_pid`'s zombie children and return it,
        so the caller can read its exit status. Returns None if the parent has no
        such zombie child (a parent may only reap its OWN children)."""
        for process in self.queue:
            if (
                process.state == ZOMBIE
                and process.parent_pid == parent_pid
                and (child_pid is None or process.pid == child_pid)
            ):
                self._remove(process.pid)
                process.state = "terminated"
                return process
        return None

    def kill_tree(self, pid: str, exit_status: int = 0) -> List[str]:
        """Terminate a process and every descendant. Deepest-first, so a child is
        already gone before its parent dies and cannot be reparented out of the
        subtree mid-sweep. Opt-in only — plain `terminate` leaves children alive."""
        if self.get(pid) is None:
            return []

        order: List[str] = []

        def collect(current: str) -> None:
            for child in self.children_of(current):
                if child.pid != current:
                    collect(child.pid)
            order.append(current)

        collect(pid)
        killed: List[str] = []
        for target in order:
            result = self.terminate(target, exit_status=exit_status)
            if result["found"]:
                killed.append(target)
        return killed

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
        if algorithm == "priority_aging":
            return priority_aging(self.queue)
        if algorithm == "mlfq":
            return mlfq(self.queue, quantums=mlfq_quantums or DEFAULT_MLFQ_QUANTUMS)
        if algorithm == "mlfq_boost":
            return mlfq_boost(self.queue, quantums=mlfq_quantums or DEFAULT_MLFQ_QUANTUMS)
        raise UnknownAlgorithmError(
            f"Unknown algorithm '{algorithm}'. Available: {', '.join(ALGORITHM_NAMES)}"
        )
