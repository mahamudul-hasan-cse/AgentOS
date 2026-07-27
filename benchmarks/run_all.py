"""Run every benchmark, print result tables, and write JSON + PNG charts.

    python -m benchmarks.run_all

Outputs land in benchmarks/results/:
    scheduler.json / memory.json   raw measurements (self-describing: each file
                                   records the parameters that produced it)
    scheduler_*.png / memory_*.png grouped bar charts, one per metric, sized for
                                   dropping straight into a report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks import memory_bench, scheduler_bench  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# colour-blind-safe qualitative palette
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

SCHEDULER_CHARTS = [
    ("avg_waiting_time", "Average waiting time", "time units (lower is better)"),
    ("avg_turnaround_time", "Average turnaround time", "time units (lower is better)"),
    ("avg_response_time", "Average response time", "time units (lower is better)"),
    ("context_switches", "Context switches", "count (lower is better)"),
    ("throughput", "Throughput", "processes per time unit (higher is better)"),
]

# metrics summarised as {mean, std, ...} across seeds -> drawn with error bars
MEMORY_CHARTS = [
    ("page_fault_rate", "Page fault rate", "faults / access (lower is better)"),
    ("hit_ratio", "Hit ratio", "hits / access (higher is better)"),
]
# metrics stored as a plain mean across seeds
MEMORY_MEAN_CHARTS = [
    ("avg_pages_in_ram_mean", "Average pages resident in RAM", "pages"),
]


def _grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    ylabel: str,
    group_labels: List[str],
    series: Dict[str, List[float]],
    errors: Dict[str, List[float]] | None = None,
) -> None:
    """One grouped bar chart: x = workload/trace, one bar per algorithm/policy.
    `errors` (if given) draws +/-1 standard deviation error bars."""
    fig, ax = plt.subplots(figsize=(9, 5))
    n_series = len(series)
    width = 0.8 / n_series
    positions = range(len(group_labels))

    for i, (name, values) in enumerate(series.items()):
        offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
        yerr = errors.get(name) if errors else None
        bars = ax.bar(
            offsets, values, width=width, label=name, color=PALETTE[i % len(PALETTE)],
            yerr=yerr, capsize=3,
            error_kw={"ecolor": "#333333", "elinewidth": 1, "capthick": 1},
        )
        ax.bar_label(bars, fmt="%.4g", fontsize=7, padding=2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    # generous pad so the bold title clears the subtitle drawn just above the axes
    ax.set_title(title, fontsize=12, fontweight="bold", pad=24)
    if subtitle:
        ax.text(
            0.5, 1.012, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color="#555555",
        )
    # headroom so bar labels (and error-bar caps) don't collide with the title
    top = max((max(v) for v in series.values() if v), default=0)
    if errors:
        top = max(
            (v + e for name, vals in series.items()
             for v, e in zip(vals, errors.get(name, [0] * len(vals)))),
            default=top,
        )
    if top > 0:
        ax.set_ylim(0, top * 1.15)
    ax.legend(frameon=False, ncols=n_series, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_scheduler_charts(results: Dict) -> List[Path]:
    profiles = list(results["profiles"])
    algorithms = scheduler_bench.ALGORITHMS
    p = results["parameters"]
    subtitle = (
        f"seed={p['seed']} · n={p['num_processes']} processes · "
        f"RR quantum={p['round_robin_quantum']} · MLFQ={p['mlfq_quantums']}"
    )

    written = []
    for metric, title, ylabel in SCHEDULER_CHARTS:
        series = {
            algo: [results["profiles"][prof]["algorithms"][algo][metric] for prof in profiles]
            for algo in algorithms
        }
        path = RESULTS_DIR / f"scheduler_{metric}.png"
        _grouped_bar_chart(path, f"Scheduler — {title}", subtitle, ylabel, profiles, series)
        written.append(path)
    return written


def write_memory_charts(results: Dict) -> List[Path]:
    traces = list(results["traces"])
    policies = memory_bench.POLICIES
    p = results["parameters"]
    subtitle = (
        f"base_seed={p['base_seed']} · {p['num_seeds']} seeds (error bars = ±1 sd) · "
        f"{p['num_pages']} pages · ~{p['pages_resident_at_once']} resident · "
        f"{p['accesses_per_trace']} accesses · "
        f"{'semantic' if p['embeddings_are_semantic'] else 'HASHING (not a fair test)'} embeddings"
    )

    written = []
    for metric, title, ylabel in MEMORY_CHARTS:
        series = {
            pol: [results["traces"][t]["policies"][pol][metric]["mean"] for t in traces]
            for pol in policies
        }
        errors = {
            pol: [results["traces"][t]["policies"][pol][metric]["std"] for t in traces]
            for pol in policies
        }
        path = RESULTS_DIR / f"memory_{metric}.png"
        _grouped_bar_chart(
            path, f"Page replacement — {title}", subtitle, ylabel, traces, series, errors
        )
        written.append(path)

    for metric, title, ylabel in MEMORY_MEAN_CHARTS:
        series = {
            pol: [results["traces"][t]["policies"][pol][metric] for t in traces]
            for pol in policies
        }
        path = RESULTS_DIR / f"memory_{metric.removesuffix('_mean')}.png"
        _grouped_bar_chart(path, f"Page replacement — {title}", subtitle, ylabel, traces, series)
        written.append(path)
    return written


def main(num_seeds: int = memory_bench.DEFAULT_NUM_SEEDS) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scheduler_results = scheduler_bench.run_benchmark()
    print(scheduler_bench.format_tables(scheduler_results))
    print()

    memory_results = memory_bench.run_benchmark(num_seeds=num_seeds)
    print(memory_bench.format_tables(memory_results))
    print()

    (RESULTS_DIR / "scheduler.json").write_text(
        json.dumps(scheduler_results, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "memory.json").write_text(
        json.dumps(memory_results, indent=2), encoding="utf-8"
    )

    charts = write_scheduler_charts(scheduler_results) + write_memory_charts(memory_results)

    print("=" * 78)
    print("ARTIFACTS")
    print("=" * 78)
    print(f"  {RESULTS_DIR / 'scheduler.json'}")
    print(f"  {RESULTS_DIR / 'memory.json'}")
    for path in charts:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="run all benchmarks")
    parser.add_argument(
        "--seeds",
        type=int,
        default=memory_bench.DEFAULT_NUM_SEEDS,
        help=(
            "seeds to average the memory benchmark over "
            f"(default {memory_bench.DEFAULT_NUM_SEEDS}; lower it for a quick run)"
        ),
    )
    raise SystemExit(main(num_seeds=parser.parse_args().seeds))
