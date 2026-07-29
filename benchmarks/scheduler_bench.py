"""Empirical comparison of the scheduling algorithms, including a starvation study.

Generates reproducible synthetic workloads (fixed seed) and runs FCFS, Round
Robin, Priority, Priority+Aging, MLFQ and MLFQ+Boost over the *identical*
workload, measuring the standard OS-textbook metrics.

Metric definitions (computed from the execution timeline):
  waiting time    = turnaround - total CPU time received
  turnaround time = completion - arrival
  response time   = first time on CPU - arrival
  throughput      = processes completed per time unit, over the makespan
  context switches= transitions between different pids in the timeline

Why MAX and PER-PRIORITY waiting time, not just the average: starvation is
invisible in an average. A scheduler that serves 20 processes instantly and
leaves 4 waiting forever still posts a respectable mean. The `starvation`
profile is built specifically to expose that, and the extra
`max_waiting_time` / `waiting_by_priority` metrics are what make it visible.

Three metrics carry the starvation argument, in increasing sharpness:
  max_waiting_time      global worst wait. Weakest: under a saturating workload
                        it grows for EVERY algorithm, so it conflates an
                        overloaded system with a starved process.
  low_priority_max_wait worst wait among priority>0 only. Isolates the victims,
                        but still credits a long process for simply being long.
  max_starvation_gap    longest stretch a process sat runnable without the CPU.
                        The only one of the three that means "the scheduler
                        passed this process over", which is what starvation is.
                        Needed for MLFQ, which ignores the priority field and
                        starves on burst length instead.

Fairness note: the scheduling algorithms mutate the Process objects they run
over (state, remaining_burst), so every algorithm is handed a freshly built
copy of the same workload rather than a shared list.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from kernel.scheduler import (
    DEFAULT_AGING_INTERVAL,
    DEFAULT_BOOST_INTERVAL,
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

# --- fixed experiment parameters (reproducible + citable) -----------------
SEED = 20260726
NUM_PROCESSES = 24
MAX_ARRIVAL = 40
NUM_PRIORITIES = 4
RR_QUANTUM = 4.0
AGING_INTERVAL = DEFAULT_AGING_INTERVAL
BOOST_INTERVAL = DEFAULT_BOOST_INTERVAL

ALGORITHMS = (
    "fcfs",
    "round_robin",
    "priority",
    "priority_aging",
    "mlfq",
    "mlfq_boost",
)

# --- shape of the `starvation` profile ------------------------------------
# A steady stream of short, top-priority arrivals whose offered load exceeds
# 1.0 (mean burst 2.5 arriving every 2.0 time units => ~1.25), so the CPU never
# runs out of priority-0 work while the stream lasts. A handful of longer,
# lower-priority processes arrive at the very start and then have to compete
# with it. Under plain Priority they cannot win, ever.
STREAM_LENGTH = 20
STREAM_INTERARRIVAL = 2.0
STREAM_BURST = (2, 3)
VICTIM_BURST = (6, 10)
VICTIMS_PER_LEVEL = 2
# stream lengths used to show that the victims' wait grows WITHOUT BOUND under
# Priority (linear in stream length) but is capped under Priority+Aging
STREAM_SWEEP = (10, 20, 40, 80)
# dials for the cost-of-the-fix sweep (both bracket the two known endpoints)
AGING_SWEEP = (2, 5, 10, 20, 40, 80, 160)
BOOST_SWEEP = (5, 10, 20, 40, 80, 160)

PROFILES: Dict[str, str] = {
    "uniform": "all bursts drawn from a narrow band (5-8): the homogeneous baseline",
    "mixed_short_long": "bimodal - 70% short jobs (1-4), 30% long jobs (15-25)",
    "heavy_tailed": "most jobs tiny (1-3), a few dominate (40-60): Pareto-like",
    "starvation": (
        "saturating stream of short priority-0 arrivals + a few long "
        "low-priority processes: built to starve the bottom of the queue"
    ),
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


def generate_starvation_workload(
    stream_length: int = STREAM_LENGTH, seed: int = SEED
) -> List[ProcessSpec]:
    """Build the starvation workload: a saturating priority-0 stream plus a few
    long, low-priority victims that arrive first and then have to compete."""
    rng = random.Random(f"{seed}-starvation-{stream_length}")
    specs: List[ProcessSpec] = []

    for i in range(stream_length):
        specs.append(
            ProcessSpec(
                pid=f"H{i:02d}",
                arrival_time=i * STREAM_INTERARRIVAL,
                burst=float(rng.randint(*STREAM_BURST)),
                priority=0,
            )
        )

    # victims at every priority level below the stream, all arriving early so
    # that nothing but the scheduler's own policy explains their waiting time
    arrival = 0.0
    for level in range(1, NUM_PRIORITIES):
        for k in range(VICTIMS_PER_LEVEL):
            specs.append(
                ProcessSpec(
                    pid=f"L{level}_{k}",
                    arrival_time=arrival,
                    burst=float(rng.randint(*VICTIM_BURST)),
                    priority=level,
                )
            )
            arrival += 1.0

    return specs


def generate_workload(profile: str, n: int = NUM_PROCESSES, seed: int = SEED) -> List[ProcessSpec]:
    """Build a reproducible workload. The seed is mixed with the profile name so
    profiles differ from each other but each is stable across runs."""
    if profile == "starvation":
        return generate_starvation_workload(seed=seed)

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
    if name == "priority_aging":
        return priority_aging(processes, aging_interval=AGING_INTERVAL)
    if name == "mlfq":
        return mlfq(processes, quantums=DEFAULT_MLFQ_QUANTUMS)
    if name == "mlfq_boost":
        return mlfq_boost(
            processes, quantums=DEFAULT_MLFQ_QUANTUMS, boost_interval=BOOST_INTERVAL
        )
    raise ValueError(f"unknown algorithm '{name}'")


def measure(specs: Sequence[ProcessSpec], timeline: Sequence[TimeSlice]) -> Dict[str, Any]:
    """Derive the textbook metrics from an execution timeline."""
    if not timeline:
        return {}

    arrival = {s.pid: s.arrival_time for s in specs}
    burst = {s.pid: s.burst for s in specs}
    priority = {s.pid: s.priority for s in specs}

    first_start: Dict[str, float] = {}
    completion: Dict[str, float] = {}
    for slice_ in timeline:
        if slice_.pid not in first_start:
            first_start[slice_.pid] = slice_.start
        first_start[slice_.pid] = min(first_start[slice_.pid], slice_.start)
        completion[slice_.pid] = max(completion.get(slice_.pid, 0.0), slice_.end)

    pids = list(completion)
    turnaround = [completion[p] - arrival[p] for p in pids]
    waiting = {p: (completion[p] - arrival[p]) - burst[p] for p in pids}
    response = [first_start[p] - arrival[p] for p in pids]

    makespan = max(s.end for s in timeline) - min(s.start for s in timeline)
    switches = sum(
        1 for a, b in zip(timeline, timeline[1:]) if a.pid != b.pid
    )

    # per-priority-level averages: the aggregate that actually reveals starvation
    by_level: Dict[int, List[float]] = {}
    for p in pids:
        by_level.setdefault(priority[p], []).append(waiting[p])
    waiting_by_priority = {
        str(level): round(statistics.fmean(vals), 3) for level, vals in sorted(by_level.items())
    }
    max_waiting_by_priority = {
        str(level): round(max(vals), 3) for level, vals in sorted(by_level.items())
    }
    worst_pid = max(waiting, key=lambda p: (waiting[p], p))

    # The headline starvation number. Deliberately NOT the global max: under a
    # saturating workload every process's wait grows, so a global max conflates
    # "the system is overloaded" with "these specific processes are starved".
    below_top = [w for p, w in waiting.items() if priority[p] > 0]

    # Longest stretch a process sat ready-but-not-running: the arrival-to-first-
    # slice gap, plus every gap between consecutive slices. This is the metric
    # that isolates starvation proper. Plain waiting time cannot: a LONG process
    # legitimately finishes last, so under MLFQ (which ignores the priority
    # field and demotes on burst length alone) a large waiting time may mean
    # nothing worse than "it had a lot of work to do". A large *gap* always
    # means the scheduler passed it over while it was runnable.
    slices_by_pid: Dict[str, List[TimeSlice]] = {}
    for slice_ in timeline:
        slices_by_pid.setdefault(slice_.pid, []).append(slice_)
    gap: Dict[str, float] = {}
    for pid, slices in slices_by_pid.items():
        ordered = sorted(slices, key=lambda s: s.start)
        spans = [ordered[0].start - arrival[pid]]
        spans += [b.start - a.end for a, b in zip(ordered, ordered[1:])]
        gap[pid] = max(spans)
    gap_by_level: Dict[int, List[float]] = {}
    for p in pids:
        gap_by_level.setdefault(priority[p], []).append(gap[p])

    return {
        "avg_waiting_time": round(statistics.fmean(waiting.values()), 3),
        "max_waiting_time": round(max(waiting.values()), 3),
        "low_priority_max_wait": round(max(below_top), 3) if below_top else 0.0,
        "max_starvation_gap": round(max(gap.values()), 3),
        "low_priority_max_gap": round(
            max((gap[p] for p in pids if priority[p] > 0), default=0.0), 3
        ),
        "avg_turnaround_time": round(statistics.fmean(turnaround), 3),
        "avg_response_time": round(statistics.fmean(response), 3),
        "throughput": round(len(pids) / makespan, 4) if makespan else 0.0,
        "context_switches": switches,
        "makespan": round(makespan, 3),
        "waiting_by_priority": waiting_by_priority,
        "max_waiting_by_priority": max_waiting_by_priority,
        "max_gap_by_priority": {
            str(level): round(max(vals), 3) for level, vals in sorted(gap_by_level.items())
        },
        "worst_waiter": {"pid": worst_pid, "priority": priority[worst_pid]},
    }


METRIC_LABELS = {
    "avg_waiting_time": "avg wait",
    "max_waiting_time": "max wait",
    "low_priority_max_wait": "max wait p>0",
    "low_priority_max_gap": "max gap p>0",
    "avg_turnaround_time": "avg turnaround",
    "avg_response_time": "avg response",
    "throughput": "throughput",
    "context_switches": "ctx switches",
    "makespan": "makespan",
}
# metrics where a LOWER value is better (used for the winner annotation)
LOWER_IS_BETTER = {
    "avg_waiting_time",
    "max_waiting_time",
    "low_priority_max_wait",
    "low_priority_max_gap",
    "avg_turnaround_time",
    "avg_response_time",
    "context_switches",
    "makespan",
}


def run_starvation_sweep() -> Dict[str, Any]:
    """Vary the length of the high-priority stream and watch what the victims'
    waiting time does. This is the part that distinguishes "a long wait" from
    "starvation": under Priority the worst wait grows linearly with the stream,
    i.e. without bound, while aging caps it at a constant."""
    sweep: Dict[str, Any] = {
        "description": (
            "worst low-priority waiting time as the priority-0 stream lengthens"
        ),
        "stream_lengths": list(STREAM_SWEEP),
        "algorithms": {},
    }
    for algorithm in ALGORITHMS:
        points = []
        for stream_length in STREAM_SWEEP:
            specs = generate_starvation_workload(stream_length=stream_length)
            metrics = measure(specs, run_algorithm(algorithm, specs))
            points.append(
                {
                    "stream_length": stream_length,
                    "low_priority_max_wait": metrics["low_priority_max_wait"],
                    "low_priority_max_gap": metrics["low_priority_max_gap"],
                    "max_waiting_time": metrics["max_waiting_time"],
                    "top_priority_avg_wait": metrics["waiting_by_priority"]["0"],
                }
            )
        sweep["algorithms"][algorithm] = points
    return sweep


def run_tradeoff_sweep() -> Dict[str, Any]:
    """Sweep the aging interval to expose what aging actually costs.

    The aging interval is a dial between two known schedulers, not a magic
    constant: as it approaches 0 every process reaches top priority immediately
    and `priority_aging` degenerates to FCFS; as it approaches infinity nothing
    ever ages and it degenerates to plain `priority_scheduling`. Sweeping it
    traces the whole fairness/priority continuum, with the two endpoints
    measured directly as reference lines.
    """
    specs = generate_starvation_workload(stream_length=STREAM_LENGTH)

    def point(label: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "setting": label,
            "low_priority_max_wait": metrics["low_priority_max_wait"],
            "low_priority_max_gap": metrics["low_priority_max_gap"],
            "top_priority_avg_wait": metrics["waiting_by_priority"]["0"],
            "avg_turnaround_time": metrics["avg_turnaround_time"],
        }

    aging_points = []
    for interval in AGING_SWEEP:
        timeline = priority_aging(
            [s.to_process() for s in specs], aging_interval=float(interval)
        )
        aging_points.append(point(str(interval), measure(specs, timeline)))

    boost_points = []
    for interval in BOOST_SWEEP:
        timeline = mlfq_boost(
            [s.to_process() for s in specs],
            quantums=DEFAULT_MLFQ_QUANTUMS,
            boost_interval=float(interval),
        )
        boost_points.append(point(str(interval), measure(specs, timeline)))

    return {
        "description": "cost of the fix: aging/boost interval vs. what it buys",
        "reference": {
            "fcfs": point("fcfs", measure(specs, run_algorithm("fcfs", specs))),
            "priority": point("priority", measure(specs, run_algorithm("priority", specs))),
            "mlfq": point("mlfq", measure(specs, run_algorithm("mlfq", specs))),
        },
        "aging_interval": aging_points,
        "mlfq_boost_interval": boost_points,
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
            "aging_interval": AGING_INTERVAL,
            "mlfq_boost_interval": BOOST_INTERVAL,
        },
        "profiles": {},
    }

    for profile, description in PROFILES.items():
        specs = generate_workload(profile)
        total_burst = sum(s.burst for s in specs)
        priority_mix: Dict[str, int] = {}
        for s in specs:
            priority_mix[str(s.priority)] = priority_mix.get(str(s.priority), 0) + 1
        profile_result = {
            "description": description,
            "workload": {
                "num_processes": len(specs),
                "total_burst": total_burst,
                "mean_burst": round(statistics.fmean([s.burst for s in specs]), 3),
                "max_burst": max(s.burst for s in specs),
                "min_burst": min(s.burst for s in specs),
                "priority_mix": dict(sorted(priority_mix.items())),
            },
            "algorithms": {},
        }
        for algorithm in ALGORITHMS:
            timeline = run_algorithm(algorithm, specs)
            profile_result["algorithms"][algorithm] = measure(specs, timeline)
        results["profiles"][profile] = profile_result

    results["starvation_sweep"] = run_starvation_sweep()
    results["tradeoff_sweep"] = run_tradeoff_sweep()
    return results


def _format_tradeoff(sweep: Dict[str, Any]) -> List[str]:
    """Render the cost-of-the-fix table: what the victims gain vs. what the
    top-priority stream pays for it."""
    lines: List[str] = ["", "--- cost of the fix: what does bounding the wait buy, and cost? ---"]
    lines.append(f"    {sweep['description']}")

    def block(title: str, points: List[Dict[str, Any]], note: str) -> None:
        lines.append("")
        lines.append(f"    {title}")
        header = ["setting", "max wait p>0", "max gap p>0", "avg wait p0", "avg turnaround"]
        rows = [
            [
                p["setting"],
                f"{p['low_priority_max_wait']:g}",
                f"{p['low_priority_max_gap']:g}",
                f"{p['top_priority_avg_wait']:g}",
                f"{p['avg_turnaround_time']:g}",
            ]
            for p in points
        ]
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        lines.append("    " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
        lines.append("    " + "  ".join("-" * x for x in widths))
        for row in rows:
            lines.append("    " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        lines.append(f"    {note}")

    ref = sweep["reference"]
    block(
        "reference points (the two schedulers aging interpolates between):",
        [ref["fcfs"], ref["priority"], ref["mlfq"]],
        "",
    )
    block(
        "priority_aging, sweeping the aging interval:",
        sweep["aging_interval"],
        "(small interval -> FCFS; large interval -> plain priority)",
    )
    block(
        "mlfq_boost, sweeping the boost interval:",
        sweep["mlfq_boost_interval"],
        "(small interval -> round-robin-ish; large interval -> plain MLFQ)",
    )
    return lines


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
    lines.append(
        f"aging interval={params['aging_interval']}  "
        f"MLFQ boost interval={params['mlfq_boost_interval']}"
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
        lines.append(
            "    priority mix: "
            + ", ".join(f"p{k}x{v}" for k, v in w["priority_mix"].items())
        )
        lines.append("")

        header = ["algorithm"] + [METRIC_LABELS[m] for m in metrics]
        rows = [
            [algo] + [f"{data['algorithms'][algo][m]:g}" for m in metrics]
            for algo in ALGORITHMS
        ]
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

        def fmt(cells, widths=widths):
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

        # per-priority breakdown: where starvation actually shows up
        levels = sorted(
            {
                lvl
                for algo in ALGORITHMS
                for lvl in data["algorithms"][algo]["waiting_by_priority"]
            },
            key=int,
        )
        lines.append("")
        lines.append("    avg waiting time by priority level (0 = highest priority):")
        p_header = ["algorithm"] + [f"p{lvl}" for lvl in levels] + ["worst waiter"]
        p_rows = []
        for algo in ALGORITHMS:
            wbp = data["algorithms"][algo]["waiting_by_priority"]
            worst = data["algorithms"][algo]["worst_waiter"]
            p_rows.append(
                [algo]
                + [f"{wbp[lvl]:g}" if lvl in wbp else "-" for lvl in levels]
                + [f"{worst['pid']} (p{worst['priority']})"]
            )
        p_widths = [max(len(h), *(len(r[i]) for r in p_rows)) for i, h in enumerate(p_header)]

        def pfmt(cells):
            return "  ".join(c.ljust(p_widths[i]) for i, c in enumerate(cells))

        lines.append("    " + pfmt(p_header))
        lines.append("    " + "  ".join("-" * x for x in p_widths))
        for row in p_rows:
            lines.append("    " + pfmt(row))

    # --- the unboundedness check -----------------------------------------
    sweep = results["starvation_sweep"]
    lines.append("")
    lines.append("--- starvation sweep: does the wait GROW, or is it bounded? ---")
    lines.append(f"    {sweep['description']}")
    lines.append("")
    s_header = ["algorithm"] + [f"stream={n}" for n in sweep["stream_lengths"]] + ["growth"]
    s_rows = []
    for algo in ALGORITHMS:
        points = sweep["algorithms"][algo]
        vals = [p["low_priority_max_wait"] for p in points]
        growth = (vals[-1] / vals[0]) if vals[0] else float("inf")
        s_rows.append([algo] + [f"{v:g}" for v in vals] + [f"x{growth:.1f}"])
    s_widths = [max(len(h), *(len(r[i]) for r in s_rows)) for i, h in enumerate(s_header)]

    def sfmt(cells):
        return "  ".join(c.ljust(s_widths[i]) for i, c in enumerate(cells))

    lines.append("    " + sfmt(s_header))
    lines.append("    " + "  ".join("-" * x for x in s_widths))
    for row in s_rows:
        lines.append("    " + sfmt(row))
    lines.append(
        "    (worst wait among priority>0 processes as the priority-0 stream"
    )
    lines.append(
        "     lengthens 8x. Growth tracking the stream means unbounded;"
    )
    lines.append(
        "     flat means the wait is capped. NOT the global max, which grows"
    )
    lines.append(
        "     for every algorithm here simply because the workload saturates.)"
    )

    lines.extend(_format_tradeoff(results["tradeoff_sweep"]))

    return "\n".join(lines)


def write_starvation_charts(results: Dict, results_dir) -> List:
    """Two charts: waiting time by priority level per algorithm (where the
    starvation is), and max wait vs stream length (that it is unbounded)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    from pathlib import Path

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#4c78a8", "#9ecae9", "#e45756", "#f58518", "#54a24b", "#88d27a"]
    written = []

    # --- chart 1: waiting time by priority level --------------------------
    data = results["profiles"]["starvation"]["algorithms"]
    levels = sorted(
        {lvl for algo in ALGORITHMS for lvl in data[algo]["waiting_by_priority"]}, key=int
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / len(ALGORITHMS)
    for i, algo in enumerate(ALGORITHMS):
        wbp = data[algo]["waiting_by_priority"]
        vals = [wbp.get(lvl, 0.0) for lvl in levels]
        offsets = [j + (i - (len(ALGORITHMS) - 1) / 2) * width for j in range(len(levels))]
        ax.bar(offsets, vals, width=width, label=algo, color=palette[i % len(palette)])
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"priority {lvl}" for lvl in levels])
    ax.set_ylabel("average waiting time (lower is better)")
    fig.suptitle("Starvation under Priority scheduling, and the effect of aging")
    ax.set_title(
        "workload: saturating stream of priority-0 arrivals + long low-priority processes",
        fontsize=8,
        color="#555555",
    )
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    path = results_dir / "scheduler_starvation_by_priority.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- chart 2: unboundedness -------------------------------------------
    sweep = results["starvation_sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    # priority and mlfq trace the identical curve here (both leave the victims
    # to the very end), so vary linestyle/marker or one hides the other exactly
    styles = ["-", "--", "-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "P"]
    for i, algo in enumerate(ALGORITHMS):
        points = sweep["algorithms"][algo]
        ax.plot(
            [p["stream_length"] for p in points],
            [p["low_priority_max_wait"] for p in points],
            marker=markers[i % len(markers)],
            linestyle=styles[i % len(styles)],
            linewidth=1.8,
            markersize=6,
            alpha=0.9,
            label=algo,
            color=palette[i % len(palette)],
        )
    ax.set_xlabel("length of the high-priority stream (number of arrivals)")
    ax.set_ylabel("worst wait among priority>0 (lower is better)")
    ax.set_title("Is the wait bounded? Low-priority wait vs. stream length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    path = results_dir / "scheduler_starvation_growth.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- chart 3: the tradeoff curve --------------------------------------
    tradeoff = results["tradeoff_sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, colour in (
        ("aging_interval", "priority_aging (aging interval)", palette[3]),
        ("mlfq_boost_interval", "mlfq_boost (boost interval)", palette[5]),
    ):
        points = tradeoff[key]
        ax.plot(
            [p["top_priority_avg_wait"] for p in points],
            [p["low_priority_max_gap"] for p in points],
            marker="o",
            label=label,
            color=colour,
        )
        seen = set()
        for p in points:
            xy = (p["top_priority_avg_wait"], p["low_priority_max_gap"])
            if xy in seen:
                continue  # the dial has saturated; one label is enough
            seen.add(xy)
            ax.annotate(
                p["setting"],
                xy,
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#555555",
            )
    for name, marker in (("fcfs", "s"), ("priority", "*"), ("mlfq", "^")):
        ref = tradeoff["reference"][name]
        ax.scatter(
            [ref["top_priority_avg_wait"]],
            [ref["low_priority_max_gap"]],
            marker=marker,
            s=110,
            zorder=5,
            label=f"{name} (reference)",
            color=palette[0],
        )
    ax.set_xlabel("avg waiting time of priority-0 stream  (the cost)")
    ax.set_ylabel("worst starvation gap among priority>0  (what you buy)")
    ax.set_title("The tradeoff: fairness bought with high-priority delay")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    path = results_dir / "scheduler_starvation_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def main() -> Dict:
    import json
    from pathlib import Path

    results = run_benchmark()
    print(format_tables(results))

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "scheduler.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    charts = write_starvation_charts(results, results_dir)
    print("")
    print("wrote:")
    print(f"  {results_dir / 'scheduler.json'}")
    for path in charts:
        print(f"  {path}")
    return results


if __name__ == "__main__":
    main()
