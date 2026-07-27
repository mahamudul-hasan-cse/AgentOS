"""Empirical comparison of the four scheduling algorithms.

Generates reproducible synthetic workloads (fixed seed) and runs FCFS, Round
Robin, Priority and MLFQ over the *identical* workload, measuring the standard
OS-textbook metrics.

Metric definitions (computed from the execution timeline):
  waiting time    = turnaround - total CPU time received
  turnaround time = completion - arrival
  response time   = first time on CPU - arrival
  throughput      = processes completed per time unit, over the makespan
  context switches= transitions between different pids in the timeline

Fairness note: the scheduling algorithms mutate the Process objects they run
over (state, remaining_burst), so every algorithm is handed a freshly built
copy of the same workload rather than a shared list.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

from kernel.scheduler import (
    DEFAULT_MLFQ_QUANTUMS,
    Process,
    TimeSlice,
    fcfs,
    mlfq,
    priority_scheduling,
    round_robin,
)

# --- fixed experiment parameters (reproducible + citable) -----------------
SEED = 20260726
NUM_PROCESSES = 24
MAX_ARRIVAL = 40
NUM_PRIORITIES = 4
RR_QUANTUM = 4.0

ALGORITHMS = ("fcfs", "round_robin", "priority", "mlfq")

PROFILES: Dict[str, str] = {
    "uniform": "all bursts drawn from a narrow band (5-8): the homogeneous baseline",
    "mixed_short_long": "bimodal - 70% short jobs (1-4), 30% long jobs (15-25)",
    "heavy_tailed": "most jobs tiny (1-3), a few dominate (40-60): Pareto-like",
}


@dataclass
class ProcessSpec:
    """An immutable workload entry. Fresh Process objects are built from these
    for every algorithm so no run can be contaminated by a previous one."""

    pid: str
    arrival_time: float
    burst: float
    priority: int

    def to_process(self) -> Process:
        return Process(
            pid=self.pid,
            arrival_time=self.arrival_time,
            estimated_burst=self.burst,
            priority=self.priority,
        )


def generate_workload(profile: str, n: int = NUM_PROCESSES, seed: int = SEED) -> List[ProcessSpec]:
    """Build a reproducible workload. The seed is mixed with the profile name so
    profiles differ from each other but each is stable across runs."""
    rng = random.Random(f"{seed}-{profile}")
    specs: List[ProcessSpec] = []
    for i in range(n):
        if profile == "uniform":
            burst = float(rng.randint(5, 8))
        elif profile == "mixed_short_long":
            burst = float(rng.randint(1, 4) if rng.random() < 0.7 else rng.randint(15, 25))
        elif profile == "heavy_tailed":
            burst = float(rng.randint(40, 60) if rng.random() < 0.12 else rng.randint(1, 3))
        else:
            raise ValueError(f"unknown profile '{profile}'")
        specs.append(
            ProcessSpec(
                pid=f"P{i:02d}",
                arrival_time=float(rng.randint(0, MAX_ARRIVAL)),
                burst=burst,
                priority=rng.randint(0, NUM_PRIORITIES - 1),
            )
        )
    return specs


def run_algorithm(name: str, specs: Sequence[ProcessSpec]) -> List[TimeSlice]:
    processes = [s.to_process() for s in specs]  # fresh copies every run
    if name == "fcfs":
        return fcfs(processes)
    if name == "round_robin":
        return round_robin(processes, quantum=RR_QUANTUM)
    if name == "priority":
        return priority_scheduling(processes)
    if name == "mlfq":
        return mlfq(processes, quantums=DEFAULT_MLFQ_QUANTUMS)
    raise ValueError(f"unknown algorithm '{name}'")


def measure(specs: Sequence[ProcessSpec], timeline: Sequence[TimeSlice]) -> Dict[str, float]:
    """Derive the textbook metrics from an execution timeline."""
    if not timeline:
        return {}

    arrival = {s.pid: s.arrival_time for s in specs}
    burst = {s.pid: s.burst for s in specs}

    first_start: Dict[str, float] = {}
    completion: Dict[str, float] = {}
    for slice_ in timeline:
        if slice_.pid not in first_start:
            first_start[slice_.pid] = slice_.start
        first_start[slice_.pid] = min(first_start[slice_.pid], slice_.start)
        completion[slice_.pid] = max(completion.get(slice_.pid, 0.0), slice_.end)

    pids = list(completion)
    turnaround = [completion[p] - arrival[p] for p in pids]
    waiting = [(completion[p] - arrival[p]) - burst[p] for p in pids]
    response = [first_start[p] - arrival[p] for p in pids]

    makespan = max(s.end for s in timeline) - min(s.start for s in timeline)
    switches = sum(
        1 for a, b in zip(timeline, timeline[1:]) if a.pid != b.pid
    )

    return {
        "avg_waiting_time": round(statistics.fmean(waiting), 3),
        "avg_turnaround_time": round(statistics.fmean(turnaround), 3),
        "avg_response_time": round(statistics.fmean(response), 3),
        "throughput": round(len(pids) / makespan, 4) if makespan else 0.0,
        "context_switches": switches,
        "makespan": round(makespan, 3),
    }


METRIC_LABELS = {
    "avg_waiting_time": "avg wait",
    "avg_turnaround_time": "avg turnaround",
    "avg_response_time": "avg response",
    "throughput": "throughput",
    "context_switches": "ctx switches",
    "makespan": "makespan",
}
# metrics where a LOWER value is better (used for the winner annotation)
LOWER_IS_BETTER = {
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "context_switches",
    "makespan",
}


def run_benchmark() -> Dict:
    """Run every algorithm over every profile. Returns a JSON-serializable dict."""
    results: Dict = {
        "benchmark": "scheduler",
        "parameters": {
            "seed": SEED,
            "num_processes": NUM_PROCESSES,
            "max_arrival_time": MAX_ARRIVAL,
            "num_priority_levels": NUM_PRIORITIES,
            "round_robin_quantum": RR_QUANTUM,
            "mlfq_quantums": list(DEFAULT_MLFQ_QUANTUMS),
        },
        "profiles": {},
    }

    for profile, description in PROFILES.items():
        specs = generate_workload(profile)
        total_burst = sum(s.burst for s in specs)
        profile_result = {
            "description": description,
            "workload": {
                "num_processes": len(specs),
                "total_burst": total_burst,
                "mean_burst": round(statistics.fmean([s.burst for s in specs]), 3),
                "max_burst": max(s.burst for s in specs),
                "min_burst": min(s.burst for s in specs),
            },
            "algorithms": {},
        }
        for algorithm in ALGORITHMS:
            timeline = run_algorithm(algorithm, specs)
            profile_result["algorithms"][algorithm] = measure(specs, timeline)
        results["profiles"][profile] = profile_result

    return results


def format_tables(results: Dict) -> str:
    """Render results as aligned, self-describing text tables."""
    lines: List[str] = []
    params = results["parameters"]
    lines.append("=" * 78)
    lines.append("SCHEDULER BENCHMARK")
    lines.append("=" * 78)
    lines.append(
        f"seed={params['seed']}  processes={params['num_processes']}  "
        f"arrivals=0..{params['max_arrival_time']}  "
        f"RR quantum={params['round_robin_quantum']}  MLFQ quantums={params['mlfq_quantums']}"
    )

    metrics = list(METRIC_LABELS)
    for profile, data in results["profiles"].items():
        w = data["workload"]
        lines.append("")
        lines.append(f"--- profile: {profile} ---")
        lines.append(f"    {data['description']}")
        lines.append(
            f"    workload: n={w['num_processes']}  total_burst={w['total_burst']:.0f}  "
            f"mean={w['mean_burst']}  range={w['min_burst']:.0f}-{w['max_burst']:.0f}"
        )
        lines.append("")

        header = ["algorithm"] + [METRIC_LABELS[m] for m in metrics]
        rows = [
            [algo] + [f"{data['algorithms'][algo][m]:g}" for m in metrics]
            for algo in ALGORITHMS
        ]
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

        def fmt(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        lines.append("    " + fmt(header))
        lines.append("    " + "  ".join("-" * x for x in widths))
        for row in rows:
            lines.append("    " + fmt(row))

        best = []
        for m in metrics:
            if m == "makespan":
                continue
            vals = {a: data["algorithms"][a][m] for a in ALGORITHMS}
            winner = (min if m in LOWER_IS_BETTER else max)(vals, key=vals.get)
            best.append(f"{METRIC_LABELS[m]}: {winner}")
        lines.append("    best -> " + " | ".join(best))

    return "\n".join(lines)


def main() -> Dict:
    results = run_benchmark()
    print(format_tables(results))
    return results


if __name__ == "__main__":
    main()
