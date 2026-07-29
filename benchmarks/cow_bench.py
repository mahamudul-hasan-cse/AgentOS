"""Measure what copy-on-write saves versus naive copy-on-fork.

A parent builds N pages, then M children fork from it. Each child issues a
seeded stream of accesses that are mostly reads with an occasional write; only a
write forces a private copy, so the interesting number is how much memory
survives being shared.

Baseline: naive copy-on-fork duplicates every parent page into every child at
spawn time, so it stores N * (M + 1) pages regardless of what anyone does. COW
stores N pages plus exactly one extra per distinct (child, page) that was
written. The gap between them is the saving.

Fixed seeds throughout, like the other benchmarks, so results are reproducible.
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from typing import Any, Dict, List

from kernel.memory import PageManager
from kernel.memory.embeddings import (
    HashingEmbedder,
    release_chroma_clients,
    set_embedder,
)

SEED = 20260727
PARENT = "parent"
PAGES_PER_PARENT = 20
TOKENS_PER_PAGE = 50
ACCESSES_PER_CHILD = 40
RAM_BUDGET_TOKENS = 20 * 50  # everything fits: this measures sharing, not eviction

#: how often a child access is a WRITE rather than a read
WRITE_RATIOS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
CHILD_COUNTS = (1, 2, 4, 8)


def build_parent(pm: PageManager, pages: int) -> None:
    for i in range(pages):
        pm.write_page(
            PARENT,
            f"p{i:03d}",
            f"parent chunk {i}: notes on scheduling, paging and syscalls",
            token_count=TOKENS_PER_PAGE,
        )


def run_case(children: int, write_ratio: float, seed: int) -> Dict[str, Any]:
    """One (children x write_ratio) cell. Returns COW vs naive accounting."""
    chroma_dir = tempfile.mkdtemp(prefix="cowbench-")
    try:
        pm = PageManager(
            ram_budget_tokens=RAM_BUDGET_TOKENS, policy="fifo", chroma_path=chroma_dir
        )
        build_parent(pm, PAGES_PER_PARENT)

        for c in range(children):
            pm.fork(PARENT, f"child{c}")

        rng = random.Random(f"{seed}-{children}-{write_ratio}")
        writes = 0
        for c in range(children):
            agent = f"child{c}"
            for _ in range(ACCESSES_PER_CHILD):
                page_id = f"p{rng.randrange(PAGES_PER_PARENT):03d}"
                if rng.random() < write_ratio:
                    pm.write_page(
                        agent, page_id, f"{agent} rewrote {page_id}", token_count=TOKENS_PER_PAGE
                    )
                    writes += 1
                else:
                    # a read never copies; touch the frame through the page table
                    pm.frame_of(agent, page_id)

        metrics = pm.cow_metrics()
        naive_pages = PAGES_PER_PARENT * (children + 1)
        naive_tokens = naive_pages * TOKENS_PER_PAGE
        cow_tokens = metrics["tokens_stored"]

        return {
            "children": children,
            "write_ratio": write_ratio,
            "writes_issued": writes,
            "cow_faults": metrics["cow_faults"],
            "frames": metrics["frames"],
            "page_table_entries": metrics["page_table_entries"],
            "cow_tokens": cow_tokens,
            "naive_tokens": naive_tokens,
            "tokens_saved": naive_tokens - cow_tokens,
            "savings_ratio": round(1 - cow_tokens / naive_tokens, 4),
        }
    finally:
        # release before rmtree: an open handle makes the delete a silent no-op
        release_chroma_clients()
        shutil.rmtree(chroma_dir, ignore_errors=True)


def run_benchmark() -> Dict[str, Any]:
    # Deterministic offline embeddings: this benchmark measures memory sharing,
    # not embedding quality, and the hashing backend keeps it fast and seeded.
    set_embedder(HashingEmbedder())
    try:
        cases: List[Dict[str, Any]] = []
        for children in CHILD_COUNTS:
            for ratio in WRITE_RATIOS:
                cases.append(run_case(children, ratio, SEED))
        return {
            "benchmark": "copy-on-write",
            "parameters": {
                "seed": SEED,
                "pages_per_parent": PAGES_PER_PARENT,
                "tokens_per_page": TOKENS_PER_PAGE,
                "accesses_per_child": ACCESSES_PER_CHILD,
                "child_counts": list(CHILD_COUNTS),
                "write_ratios": list(WRITE_RATIOS),
                "baseline": "naive copy-on-fork duplicates every parent page per child",
            },
            "cases": cases,
        }
    finally:
        set_embedder(None)


def format_tables(results: Dict[str, Any]) -> str:
    p = results["parameters"]
    lines = [
        "=" * 84,
        "COPY-ON-WRITE BENCHMARK",
        "=" * 84,
        f"seed={p['seed']}  parent pages={p['pages_per_parent']} x "
        f"{p['tokens_per_page']} tokens  accesses/child={p['accesses_per_child']}",
        f"baseline: {p['baseline']}",
    ]

    header = [
        "children", "write%", "writes", "cow faults", "frames",
        "cow tokens", "naive tokens", "saved", "saved %",
    ]
    rows = []
    for c in results["cases"]:
        rows.append([
            str(c["children"]),
            f"{c['write_ratio'] * 100:g}",
            str(c["writes_issued"]),
            str(c["cow_faults"]),
            str(c["frames"]),
            str(c["cow_tokens"]),
            str(c["naive_tokens"]),
            str(c["tokens_saved"]),
            f"{c['savings_ratio'] * 100:.1f}",
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def fmt(cells):
        return "  ".join(c.rjust(widths[i]) for i, c in enumerate(cells))

    lines.append("")
    lines.append("    " + fmt(header))
    lines.append("    " + "  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("    " + fmt(row))

    read_only = [c for c in results["cases"] if c["write_ratio"] == 0.0]
    write_all = [c for c in results["cases"] if c["write_ratio"] == 1.0]
    lines.append("")
    lines.append(
        "    read-only children  -> mean saving "
        f"{statistics.fmean(c['savings_ratio'] for c in read_only) * 100:.1f}%"
        "  (COW is pure win: nothing is ever copied)"
    )
    lines.append(
        "    write-everything    -> mean saving "
        f"{statistics.fmean(c['savings_ratio'] for c in write_all) * 100:.1f}%"
        "  (COW degrades toward the naive baseline, as it must)"
    )
    return "\n".join(lines)


def main() -> Dict[str, Any]:
    results = run_benchmark()
    print(format_tables(results))
    return results


if __name__ == "__main__":
    main()
