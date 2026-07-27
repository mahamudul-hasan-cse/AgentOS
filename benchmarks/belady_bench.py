"""Belady's Anomaly: sweep RAM capacity and look for MORE frames causing MORE faults.

Intuitively, giving a page-replacement policy more memory should never make it
worse. For some policies that intuition is provably right; for others it is
simply false, and FIFO is the textbook counterexample.

THE THEORY THIS TESTS AGAINST
-----------------------------
* **LRU is a stack algorithm.** The set of pages resident in N frames is always
  a subset of the set resident in N+1 frames, for every prefix of every
  reference string. That inclusion property makes LRU *provably immune* to the
  anomaly. **If this benchmark reports an anomaly under LRU, that is a bug in
  our implementation, not a discovery** — it is the reason LRU is included here
  at all, as a self-check on the rest of the numbers.
* **FIFO is not a stack algorithm** and is known to exhibit the anomaly; the
  canonical demonstration is reproduced below.
* **Semantic-LRU has no stack property.** It evicts by embedding distance from
  the current query, not by recency, so nothing guarantees the inclusion
  property holds. Whether it exhibits the anomaly is genuinely open, and is the
  question this benchmark exists to answer about our own algorithm.
* **Random likewise has no stack guarantee** and serves as the control.

Two experiments:

1. `classic_reference_string()` — the textbook string 1,2,3,4,1,2,5,1,2,3,4,5 at
   3 vs 4 frames. Deterministic, no seeding. Memory starts COLD (every page in
   swap, nothing resident) so the fault counts are directly comparable with the
   published numbers: FIFO 9 -> 10 (the anomaly), LRU 10 -> 8 (immune).

2. `sweep()` — every policy over every trace from memory_bench, across a range
   of capacities, multi-seeded like the other benchmarks, reporting the mean
   fault rate at each capacity and flagging any capacity step where it rises.

An anomaly is only reported with the number of seeds it held on, so one noisy
seed is visible as such rather than being presented as a result.
"""

from __future__ import annotations

import math
import random
import statistics
import tempfile
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

from kernel.memory.embeddings import set_embedder

from benchmarks.memory_bench import (
    AGENT,
    PARAPHRASE_TRACES,
    POLICIES,
    TOKENS_PER_PAGE,
    TRACES,
    BenchPageManager,
    build_pages,
    query_for,
    release_chroma_clients,
    release_chroma_dirs,
    select_embedder,
)

SEED = 20260727
DEFAULT_NUM_SEEDS = 10
DEFAULT_MIN_FRAMES = 2
DEFAULT_MAX_FRAMES = 10
#: shorter than memory_bench's 120: this sweep multiplies runs by capacity too
ACCESSES = 60

#: the canonical minimal demonstration from the OS literature
CLASSIC_STRING = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
CLASSIC_CAPACITIES = (3, 4)
#: published fault counts for the string above, used as a correctness anchor
CLASSIC_EXPECTED = {"fifo": {3: 9, 4: 10}, "lru": {3: 10, 4: 8}}
#: policies whose fault counts are reproducible across implementations;
#: `random` is excluded because a seeded RNG is consumed in a different
#: order by each implementation.
DETERMINISTIC_POLICIES = ("fifo", "lru", "semantic_lru")


def _make_manager(frames: int, policy: str, chroma_dir: str, seed: int) -> BenchPageManager:
    return BenchPageManager(
        ram_budget_tokens=frames * TOKENS_PER_PAGE,
        policy=policy,
        chroma_path=chroma_dir,
        rng=random.Random(f"{seed}-evict-{policy}-{frames}"),
    )


def _cold_start(manager: BenchPageManager, agent: str) -> None:
    """Evict everything so memory starts empty, as a textbook trace assumes.

    Priming writes each page, which leaves the last `frames` of them resident.
    A reference string is defined over an empty memory, so without this the
    first few accesses would be free hits and the counts would not line up with
    the published ones."""
    for page_id in list(manager.ram[agent].keys()):
        manager._evict(agent, page_id)


# --- experiment 1: the canonical demonstration ---------------------------


def classic_reference_string() -> Dict[str, Any]:
    pages = [
        {"page_id": f"P{n}", "topic": "classic", "content": f"reference page number {n}"}
        for n in sorted(set(CLASSIC_STRING))
    ]
    by_number = {int(p["page_id"][1:]): p for p in pages}

    per_policy: Dict[str, Dict[int, int]] = {}
    dirs: List[str] = []
    for policy in POLICIES:
        per_capacity: Dict[int, int] = {}
        for frames in CLASSIC_CAPACITIES:
            chroma_dir = tempfile.mkdtemp(prefix=f"belady-classic-{policy}-")
            dirs.append(chroma_dir)
            manager = _make_manager(frames, policy, chroma_dir, SEED)
            for page in pages:
                manager.write_page(
                    AGENT, page["page_id"], page["content"], token_count=TOKENS_PER_PAGE
                )
            _cold_start(manager, AGENT)

            faults = 0
            for number in CLASSIC_STRING:
                page = by_number[number]
                if manager.read(AGENT, page["content"]).page_fault:
                    faults += 1
            per_capacity[frames] = faults
            release_chroma_clients()
        per_policy[policy] = per_capacity

    release_chroma_dirs(dirs)

    low, high = CLASSIC_CAPACITIES
    return {
        "reference_string": CLASSIC_STRING,
        "capacities": list(CLASSIC_CAPACITIES),
        "faults": per_policy,
        "anomalies": {
            policy: counts[high] - counts[low]
            for policy, counts in per_policy.items()
            if counts[high] > counts[low]
        },
        "matches_published": {
            policy: {
                str(cap): per_policy[policy][cap] == expected[cap]
                for cap in CLASSIC_CAPACITIES
            }
            for policy, expected in CLASSIC_EXPECTED.items()
        },
    }


# --- experiment 2: broad capacity sweep ----------------------------------


class PolicySim:
    """An in-memory replica of PageManager's replacement policies.

    WHY A REPLICA. Driving the sweep through the real PageManager costs ~8s per
    cell, essentially all of it ChromaDB persistent writes (~25 ops/sec), and
    the sweep has hundreds of cells — hours of vector-store I/O to measure
    something that is purely a property of victim selection. This class replays
    a reference string over the same four policies with no vector store at all.

    IT IS NOT TAKEN ON TRUST. `validate_against_page_manager()` replays the
    canonical string through both this simulator and the real PageManager and
    asserts identical fault counts for every policy at every capacity; the
    result is reported alongside the numbers. If they ever diverge, the sweep's
    conclusions do not transfer and the report says so.

    Each policy mirrors the kernel exactly:
      fifo         -> first key of the resident OrderedDict (insertion order,
                      unchanged by a hit), as in PageManager.ram
      lru          -> smallest last-access stamp, as in Page.last_accessed
      random       -> BenchPageManager's seeded rng.choice over resident keys
      semantic_lru -> lowest cosine similarity to the current query embedding,
                      which is what ChromaDB's farthest-ranked result means
    """

    def __init__(
        self,
        frames: int,
        policy: str,
        rng: random.Random,
        embeddings: Dict[str, Sequence[float]],
    ) -> None:
        self.capacity = frames
        self.policy = policy
        self.rng = rng
        self.embeddings = embeddings
        self.resident: "OrderedDict[str, int]" = OrderedDict()
        self.clock = 0

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def _victim(self, query_embedding: Sequence[float]) -> str:
        if self.policy == "fifo":
            return next(iter(self.resident))
        if self.policy == "lru":
            return min(self.resident, key=lambda p: self.resident[p])
        if self.policy == "random":
            return self.rng.choice(list(self.resident))
        return min(
            self.resident,
            key=lambda p: self._cosine(self.embeddings.get(p, ()), query_embedding),
        )

    def access(self, page_id: str, query_embedding: Sequence[float]) -> bool:
        """Reference a page; True if it faulted."""
        self.clock += 1
        if page_id in self.resident:
            # a hit refreshes LRU recency but must NOT reorder FIFO
            self.resident[page_id] = self.clock
            return False
        if len(self.resident) >= self.capacity:
            del self.resident[self._victim(query_embedding)]
        self.resident[page_id] = self.clock
        return True


def validate_against_page_manager() -> Dict[str, Any]:
    """Replay the canonical string through the real PageManager and through
    PolicySim, and check they agree for every policy at both capacities."""
    from kernel.memory.embeddings import embed_text

    pages = [
        {"page_id": f"P{n}", "content": f"reference page number {n}"}
        for n in sorted(set(CLASSIC_STRING))
    ]
    embeddings = {p["page_id"]: embed_text(p["content"]) for p in pages}
    by_number = {int(p["page_id"][1:]): p for p in pages}

    agreement: Dict[str, Dict[str, Any]] = {}
    for policy in POLICIES:
        per_capacity = {}
        for frames in CLASSIC_CAPACITIES:
            sim = PolicySim(
                frames, policy, random.Random(f"{SEED}-evict-{policy}-{frames}"), embeddings
            )
            faults = 0
            for number in CLASSIC_STRING:
                page = by_number[number]
                if sim.access(page["page_id"], embeddings[page["page_id"]]):
                    faults += 1
            per_capacity[frames] = faults
        agreement[policy] = per_capacity
    return agreement


def run_one(
    policy: str, frames: int, pages: Sequence[dict], trace: Sequence[int], paraphrased: bool,
    seed: int, embeddings: Dict[str, Sequence[float]],
    query_embeddings: Dict[str, Sequence[float]],
) -> float:
    """Replay a trace at a given capacity; return the page-fault rate."""
    sim = PolicySim(
        frames, policy, random.Random(f"{seed}-evict-{policy}-{frames}"), embeddings
    )
    faults = 0
    for idx in trace:
        page = pages[idx]
        pid = page["page_id"]
        if sim.access(pid, query_embeddings[pid] if paraphrased else embeddings[pid]):
            faults += 1
    return faults / len(trace)


def sweep(
    num_seeds: int = DEFAULT_NUM_SEEDS,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Dict[str, Any]:
    from kernel.memory.embeddings import embed_text

    from benchmarks.memory_bench import PARAPHRASE_QUERIES

    seeds = [SEED + i for i in range(num_seeds)]
    capacities = list(range(min_frames, max_frames + 1))
    traces_out: Dict[str, Any] = {}

    for trace_name, generator in TRACES.items():
        paraphrased = trace_name in PARAPHRASE_TRACES
        policy_curves: Dict[str, Any] = {}
        for policy in POLICIES:
            per_capacity: Dict[int, List[float]] = {c: [] for c in capacities}
            for seed in seeds:
                pages = build_pages(seed)
                trace = generator(pages, ACCESSES, seed)
                embeddings = {p["page_id"]: embed_text(p["content"]) for p in pages}
                query_embeddings = {
                    p["page_id"]: embed_text(PARAPHRASE_QUERIES[p["page_id"]])
                    for p in pages
                } if paraphrased else embeddings
                for frames in capacities:
                    per_capacity[frames].append(
                        run_one(
                            policy, frames, pages, trace, paraphrased, seed,
                            embeddings, query_embeddings,
                        )
                    )
            policy_curves[policy] = {
                "capacities": capacities,
                "mean_fault_rate": [
                    round(statistics.fmean(per_capacity[c]), 4) if per_capacity[c] else None
                    for c in capacities
                ],
                "per_seed": {str(c): per_capacity[c] for c in capacities},
            }
        traces_out[trace_name] = {"policies": policy_curves}

    return {
        "seeds": seeds,
        "capacities": capacities,
        "accesses_per_trace": ACCESSES,
        "traces": traces_out,
    }


# --- anomaly detection ---------------------------------------------------


def detect_anomalies(sweep_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every capacity step where the MEAN fault rate rises.

    Also counts how many individual seeds moved the same way, so a step driven
    by one outlier is distinguishable from a systematic effect."""
    found: List[Dict[str, Any]] = []
    for trace_name, trace_data in sweep_results["traces"].items():
        for policy, curve in trace_data["policies"].items():
            caps = curve["capacities"]
            means = curve["mean_fault_rate"]
            for i in range(len(caps) - 1):
                low, high = caps[i], caps[i + 1]
                a, b = means[i], means[i + 1]
                if a is None or b is None or b <= a:
                    continue
                seeds_low = curve["per_seed"][str(low)]
                seeds_high = curve["per_seed"][str(high)]
                pairs = list(zip(seeds_low, seeds_high))
                seeds_worse = sum(1 for x, y in pairs if y > x)
                found.append(
                    {
                        "trace": trace_name,
                        "policy": policy,
                        "frames": [low, high],
                        "fault_rate": [a, b],
                        "magnitude": round(b - a, 4),
                        "seeds_worse": seeds_worse,
                        "seeds_total": len(pairs),
                        # a step is only called systematic when a clear
                        # majority of seeds moved the same way; 3/5 is barely
                        # above chance and is reported as weak instead
                        "systematic": len(pairs) > 0 and seeds_worse >= 0.75 * len(pairs),
                        "weak": len(pairs) > 0
                        and len(pairs) / 2 < seeds_worse < 0.75 * len(pairs),
                    }
                )
    return found



# --- charts --------------------------------------------------------------

RESULTS_DIR = __import__("pathlib").Path(__file__).resolve().parent / "results"
PALETTE = {"fifo": "#4c78a8", "lru": "#f58518", "semantic_lru": "#54a24b", "random": "#e45756"}
#: FIFO and LRU produce identical curves on several traces (neither looks at
#: the query), so distinct dash patterns and widths keep a hidden line visible
#: underneath the one drawn on top of it.
STYLES = {
    "fifo": {"linestyle": "-", "linewidth": 3.2, "marker": "o", "alpha": 0.9},
    "lru": {"linestyle": (0, (6, 3)), "linewidth": 1.8, "marker": "s"},
    "semantic_lru": {"linestyle": (0, (2, 2)), "linewidth": 1.8, "marker": "^"},
    "random": {"linestyle": (0, (1, 2)), "linewidth": 1.8, "marker": "d"},
}


def write_charts(results: Dict[str, Any]) -> List[str]:
    """One chart per trace: fault rate vs capacity, a line per policy, with any
    upward (anomalous) segment marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    anomalies = results["anomalies"]
    written: List[str] = []

    for trace_name, trace_data in results["sweep"]["traces"].items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for policy, curve in trace_data["policies"].items():
            caps = curve["capacities"]
            means = curve["mean_fault_rate"]
            style = dict(STYLES.get(policy, {}))
            ax.plot(caps, means, markersize=4, label=policy,
                    color=PALETTE.get(policy, "#888888"), **style)
            # mark every segment where MORE frames gave MORE faults
            for a in anomalies:
                if a["trace"] != trace_name or a["policy"] != policy:
                    continue
                lo, hi = a["frames"]
                y0, y1 = a["fault_rate"]
                ax.annotate(
                    "", xy=(hi, y1), xytext=(lo, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2),
                )
                ax.scatter([hi], [y1], s=90, facecolors="none",
                           edgecolors="#d62728", linewidths=2, zorder=5)
                ax.annotate(
                    f"anomaly +{a['magnitude']:.3f}\n"
                    f"{a['seeds_worse']}/{a['seeds_total']} seeds",
                    xy=(hi, y1), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=7, color="#d62728",
                )

        ax.set_xlabel("frames (RAM capacity, pages)")
        ax.set_ylabel("mean page fault rate (lower is better)")
        ax.set_title(f"Belady sweep - {trace_name}", fontsize=12, fontweight="bold", pad=24)
        ax.text(0.5, 1.012,
                f"{len(results['sweep']['seeds'])} seeds - red arrow = MORE frames, "
                "MORE faults (Belady's Anomaly)",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#555555")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.11))
        fig.tight_layout()
        path = RESULTS_DIR / f"belady_{trace_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


# --- reporting -----------------------------------------------------------


def format_tables(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    classic = results["classic"]
    lines.append("=" * 80)
    lines.append("BELADY'S ANOMALY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. CANONICAL REFERENCE STRING " + str(classic["reference_string"]))
    lines.append("   (cold start, deterministic; published: FIFO 9->10, LRU 10->8)")
    lines.append("")
    caps = classic["capacities"]
    header = ["policy"] + [f"{c} frames" for c in caps] + ["delta", "anomaly?"]
    rows = []
    for policy, counts in classic["faults"].items():
        delta = counts[caps[-1]] - counts[caps[0]]
        rows.append(
            [policy]
            + [str(counts[c]) for c in caps]
            + [f"{delta:+d}", "YES" if delta > 0 else "no"]
        )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def fmt(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines.append("    " + fmt(header))
    lines.append("    " + "  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("    " + fmt(row))

    lines.append("")
    for policy, checks in classic["matches_published"].items():
        ok = all(checks.values())
        lines.append(
            f"    {policy} matches the published counts: {'YES' if ok else 'NO -> ' + str(checks)}"
        )

    agrees = classic.get("simulator_agrees", {})
    if agrees:
        all_agree = all(agrees.values())
        lines.append("")
        lines.append(
            "    sweep simulator reproduces the real PageManager (deterministic "
            "policies): " + ("YES" if all_agree else f"NO -> {agrees}")
        )
        sim = classic.get("simulator_faults", {}).get("random", {})
        real = classic["faults"].get("random", {})
        if sim and real:
            lines.append(
                f"      (random excluded from the check: seeded RNG, real "
                f"{real[caps[0]]}/{real[caps[1]]} vs sim {sim[caps[0]]}/{sim[caps[1]]}"
                " - draw order differs between implementations, not semantics)"
            )
        if not all_agree:
            lines.append(
                "      !! the sweep below therefore does NOT necessarily describe the"
            )
            lines.append("         real implementation - treat it as unvalidated")

    # 2. sweep
    sweep_results = results["sweep"]
    lines.append("")
    lines.append("=" * 80)
    lines.append(
        f"2. CAPACITY SWEEP  frames {sweep_results['capacities'][0]}"
        f"-{sweep_results['capacities'][-1]}, {len(sweep_results['seeds'])} seeds, "
        f"{sweep_results['accesses_per_trace']} accesses"
    )
    lines.append("=" * 80)
    for trace_name, trace_data in sweep_results["traces"].items():
        lines.append("")
        lines.append(f"--- trace: {trace_name} --- (mean fault rate by frames)")
        caps = sweep_results["capacities"]
        header = ["policy"] + [str(c) for c in caps]
        rows = []
        for policy, curve in trace_data["policies"].items():
            rows.append(
                [policy]
                + [
                    "n/a" if v is None else f"{v:.3f}"
                    for v in curve["mean_fault_rate"]
                ]
            )
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        lines.append("    " + fmt(header))
        lines.append("    " + "  ".join("-" * w for w in widths))
        for row in rows:
            lines.append("    " + fmt(row))

    # 3. anomalies
    anomalies = results["anomalies"]
    lines.append("")
    lines.append("=" * 80)
    lines.append("3. ANOMALIES DETECTED (more frames -> higher mean fault rate)")
    lines.append("=" * 80)
    if not anomalies:
        lines.append("    none")
    else:
        header = ["trace", "policy", "frames", "fault rate", "magnitude", "seeds", "systematic"]
        rows = []
        for a in anomalies:
            rows.append([
                a["trace"], a["policy"],
                f"{a['frames'][0]}->{a['frames'][1]}",
                f"{a['fault_rate'][0]:.3f}->{a['fault_rate'][1]:.3f}",
                f"+{a['magnitude']:.4f}",
                f"{a['seeds_worse']}/{a['seeds_total']}",
                "yes" if a["systematic"] else "no (noise?)",
            ])
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        lines.append("    " + fmt(header))
        lines.append("    " + "  ".join("-" * w for w in widths))
        for row in rows:
            lines.append("    " + fmt(row))

    lru_bugs = [a for a in anomalies if a["policy"] == "lru"]
    lines.append("")
    if lru_bugs:
        lines.append(
            "    !! LRU ANOMALY REPORTED -> LRU is a stack algorithm and is provably"
        )
        lines.append(
            "       immune. This indicates an IMPLEMENTATION BUG, not a discovery:"
        )
        for a in lru_bugs:
            lines.append(
                f"         {a['trace']} {a['frames'][0]}->{a['frames'][1]} "
                f"+{a['magnitude']:.4f} on {a['seeds_worse']}/{a['seeds_total']} seeds"
            )
    else:
        lines.append("    LRU: no anomaly at any capacity step (consistent with the")
        lines.append("         stack-algorithm proof - a good sign the harness is sound)")

    for policy in ("fifo", "semantic_lru", "random"):
        hits = [a for a in anomalies if a["policy"] == policy and a["systematic"]]
        if hits:
            worst = max(hits, key=lambda a: a["magnitude"])
            lines.append(
                f"    {policy}: {len(hits)} systematic anomaly step(s); worst "
                f"{worst['trace']} {worst['frames'][0]}->{worst['frames'][1]} "
                f"+{worst['magnitude']:.4f} ({worst['seeds_worse']}/{worst['seeds_total']} seeds)"
            )
        else:
            lines.append(f"    {policy}: no systematic anomaly")

    return "\n".join(lines)


def run_benchmark(
    num_seeds: int = DEFAULT_NUM_SEEDS,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Dict[str, Any]:
    embedder = select_embedder()
    set_embedder(embedder)
    try:
        classic = classic_reference_string()
        # Cross-check the fast simulator against the real PageManager. Only the
        # DETERMINISTIC policies are comparable: `random` draws from a seeded
        # RNG, and the two implementations consume draws in a different order,
        # so a mismatch there says nothing about replacement semantics.
        sim_counts = validate_against_page_manager()
        classic["simulator_agrees"] = {
            policy: sim_counts[policy] == classic["faults"][policy]
            for policy in DETERMINISTIC_POLICIES
        }
        classic["simulator_faults"] = sim_counts
        sweep_results = sweep(num_seeds, min_frames, max_frames)
        return {
            "benchmark": "belady",
            "parameters": {
                "seed": SEED,
                "num_seeds": num_seeds,
                "capacities": sweep_results["capacities"],
                "accesses_per_trace": ACCESSES,
                "tokens_per_page": TOKENS_PER_PAGE,
                "embedder": embedder.describe(),
            },
            "classic": classic,
            "sweep": sweep_results,
            "anomalies": detect_anomalies(sweep_results),
        }
    finally:
        set_embedder(None)


def main(num_seeds: int = DEFAULT_NUM_SEEDS, max_frames: int = DEFAULT_MAX_FRAMES) -> Dict[str, Any]:
    import json

    results = run_benchmark(num_seeds=num_seeds, max_frames=max_frames)
    print(format_tables(results))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "belady.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    charts = write_charts(results)
    print()
    print("=" * 80)
    print("ARTIFACTS")
    print("=" * 80)
    print(f"  {RESULTS_DIR / 'belady.json'}")
    for c in charts:
        print(f"  {c}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Belady's Anomaly sweep")
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ns = parser.parse_args()
    main(num_seeds=ns.seeds, max_frames=ns.max_frames)
