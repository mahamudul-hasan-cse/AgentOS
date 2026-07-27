# Benchmarks

An empirical evaluation of the algorithms implemented in this kernel, producing
the comparative metrics an OS textbook uses. Everything is seeded, so results
are reproducible and citable — re-running produces byte-identical JSON.

## Running

From the project root, with the venv active:

```bash
python -m benchmarks.run_all             # both suites + JSON + PNG charts
python -m benchmarks.run_all --seeds 3    # quicker run, fewer seeds
python -m benchmarks.scheduler_bench      # scheduler tables only
python -m benchmarks.memory_bench --seeds 10   # page-replacement only
```

**Runtime.** The memory benchmark repeats every trace × policy across
`--seeds` seeds (default 10), which is 200 full trace replays (4 policies) and takes roughly
**45–55 minutes** against a local Ollama. Use `--seeds 2` or `--seeds 3` for a
fast sanity run; the scheduler benchmark is near-instant.

Artifacts land in `benchmarks/results/`:

- `scheduler.json`, `memory.json` — raw measurements. Each file embeds the
  parameters that produced it, so a result is self-describing.
- `scheduler_*.png`, `memory_*.png` — grouped bar charts, one per metric,
  ready to drop into a report.

For the memory suite, start Ollama and `ollama pull nomic-embed-text` first.
Without it the run falls back to hashing embeddings and prints a warning —
Semantic-LRU under hashing is **not** a fair test of it.

## 1. Scheduler benchmark

24 processes with seeded arrival times, bursts and priorities, run through all
four algorithms over the **identical** workload. (The algorithms mutate the
process objects they schedule, so each run is handed freshly built copies.)

Three workload profiles:

| profile | shape |
|---|---|
| `uniform` | all bursts in a narrow 5–8 band — the homogeneous baseline |
| `mixed_short_long` | bimodal: 70% short (1–4), 30% long (15–25) |
| `heavy_tailed` | most jobs tiny (1–3), ~12% enormous (40–60) |

### Metrics

| metric | definition | direction |
|---|---|---|
| avg waiting time | turnaround − CPU time actually received | lower better |
| avg turnaround time | completion − arrival | lower better |
| avg response time | first moment on CPU − arrival | lower better |
| throughput | processes completed per time unit, over the makespan | higher better |
| context switches | transitions between different pids in the timeline | lower better |

### How to read the results

- **Waiting/turnaround** measure overall efficiency; **response time** measures
  interactivity — how fast a process gets *any* service. A scheduler can be
  excellent at one and poor at the other, which is the whole point of comparing.
- **Throughput is identical across all four algorithms here, and that is
  expected, not a bug.** All four are work-conserving on a single CPU over the
  same total burst with no idle gaps, so the makespan — and therefore
  throughput — is fixed by the workload, not the policy. It is reported for
  completeness; it simply does not discriminate in this setting.
- **Context switches** are the cost side of preemption: RR and MLFQ buy better
  response time by paying more switches (a real overhead this simulation does
  not otherwise charge for).

## 2. Memory / page-replacement benchmark

A universe of 20 pages (4 topics × 5 topically-related sentences), exercised by
five access patterns. RAM holds ~5 pages. Each policy gets a fresh `PageManager`
with its own ChromaDB directory, primed by writing every page, after which only
the trace's reads are measured.

### Multi-seed methodology

Every trace × policy combination is repeated across **N seeds** (default 10) and
reported as **mean ± sample standard deviation**. Per seed the following vary:

- **the corpus**: page order is shuffled, which changes the priming order and
  therefore which pages start resident in RAM, and changes which pages the
  order-driven `sequential` and `looping` traces walk;
- **the access sequence**: `random`, `clustered` and `paraphrased` orders are
  regenerated from the seed.

The page_id → content → paraphrase mapping is never disturbed, so the
hand-written paraphrase queries stay valid; only the *ordering* varies for that
trace, exactly as intended.

All four policies see the identical corpus and access order within a seed. That
matters: it means seed-to-seed variance is *common-mode*, so the benchmark also
reports a **paired** comparison (per-seed differences) alongside the raw means.
Two independent mean ± sd summaries can overlap heavily even when one policy
wins on every single seed, so both views are printed and neither is quoted
alone.

An access is driven by querying with a page's own content, so the intended page
is the exact-match nearest neighbour. `retr. acc` (retrieval accuracy) is a
sanity metric: it should be 1.0, confirming each access really did retrieve the
page the trace asked for.

| trace | pattern |
|---|---|
| `sequential` | walks all 20 pages in order, repeatedly |
| `random` | uniformly random page ids |
| `looping` | cycles over 8 pages > RAM capacity — the classic pathological case for LRU and FIFO |
| `clustered` | dwells within one topic, then switches — semantic locality |
| `paraphrased` | **same access order as `clustered`**, but each access is driven by a hand-written paraphrase instead of the page's own text |

#### The `paraphrased` trace

This is the trace that actually tests the semantic claim. The 20 paraphrase
queries in `PARAPHRASE_QUERIES` are **hand-written** (not generated), stored in
`memory_bench.py` so the trace is reproducible, and each asks for its page's
idea in different words — e.g. the page *"Green plants convert sunlight into
chemical energy stored as glucose"* is accessed via *"how vegetation makes fuel
from daylight"*.

Because it reuses `clustered`'s access order exactly, the two traces form a
clean A/B where the only variable is whether the query lexically matches the
page.

The benchmark **measures** the "minimal word overlap" claim rather than
asserting it, printing a `paraphrase_lexical_baseline` block:

- mean Jaccard overlap between query and page text: **0.0094** (max 0.0667),
  ~0.15 shared words per pair
- **lexical-only top-1 accuracy: 0.0** — a pure word-overlap matcher (the
  hashing embedder) ranks the correct page first for *none* of the 20 queries

So the trace genuinely cannot be solved by vocabulary matching. Recency
(LRU/FIFO) gets no signal from the query at all.

### Metrics

| metric | definition | direction |
|---|---|---|
| page fault rate | faults / accesses | lower better |
| hit ratio | 1 − fault rate | higher better |
| avg pages in RAM | mean resident pages across the trace | — (capacity check) |
| retr. acc | fraction of accesses returning the *exact* intended page | higher better |
| topic match | fraction returning *any* page from the right topic | higher better |
| distinct pages | distinct pages actually retrieved over the trace | — (confound check) |

`avg pages in RAM` sits at the capacity ceiling (5) for every policy once
primed; it confirms all policies are being compared at equal memory pressure
rather than one quietly using less RAM.

`retr. acc` is 1.0 for the exact-text traces by construction. On `paraphrased`
it drops to ~0.61 — but `topic match` stays at **1.0**, i.e. every paraphrase
landed in the correct topic and the misses are all sibling pages within that
topic. Note `retr. acc` is identical across all four policies, as it must be:
which page is *nearest* is a property of the embeddings, not of the replacement
policy.

`distinct pages` exists to stop a misleading comparison. Under paraphrase,
several queries collapse onto the same nearest page, so the trace touches only
fewer distinct pages than `clustered` does — a smaller effective working set.
That alone lowers fault rates for reasons unrelated to any policy, so
**`paraphrased` and `clustered` fault rates must not be compared to each other**.
Comparisons *between policies within* a trace remain valid, which is what the
experiment is for.

### The Random-eviction control

A fourth policy, **`random`** (evict a uniformly random resident page, from a
per-seed RNG so it stays reproducible), is included purely as a scientific
control. Semantic-LRU's robust wins over LRU are on traces where recency is
pathological, and *any* policy that breaks recency ordering should do well
there. Random eviction is therefore the null hypothesis for "the embeddings are
doing real work": if Random matches Semantic-LRU, the embeddings contribute
nothing measurable.

It lives **only in this benchmark** — implemented as `BenchPageManager`, a
subclass that overrides `_make_room` (the single place `PageManager` chooses a
victim, used by both writes and page-fault reloads) and delegates every other
policy untouched. It was deliberately **not** added to
`kernel/memory/replacement.py`: it is a control, not something anyone should run
in production.

## Findings (10 seeds, 4 policies — as measured, not tuned)

Page fault rate, mean ± sd across 10 seeds (lower is better; **bold** = best):

| trace | FIFO | LRU | Semantic-LRU | Random |
|---|---|---|---|---|
| `sequential` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **0.9400 ± 0.0179** | 0.9858 ± 0.0097 |
| `random` | 0.7542 ± 0.0458 | 0.7642 ± 0.0497 | 0.7517 ± 0.0351 | **0.7450 ± 0.0488** |
| `looping` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8008 ± 0.0794 | **0.6875 ± 0.0258** |
| `clustered` | 0.4800 ± 0.0465 | 0.4725 ± 0.0470 | **0.4583 ± 0.0441** | 0.5342 ± 0.0348 |
| `paraphrased` | 0.3808 ± 0.0401 | **0.3717 ± 0.0431** | 0.3733 ± 0.0442 | 0.4125 ± 0.0424 |

Semantic-LRU against each baseline (paired across seeds):

| trace | vs LRU | vs Random (the control) |
|---|---|---|
| `sequential` | +0.0600, 10/10 — **robust** | +0.0458, 10/10 — **robust: embeddings help** |
| `random` | +0.0125, 7/10 — within noise | −0.0067, 4/10 — **no benefit** |
| `looping` | +0.1992, 10/10 — **robust** | **−0.1133, 1/10 — Random is far better** |
| `clustered` | +0.0141, 7/10 — within noise | +0.0758, 10/10 — **robust: embeddings help** |
| `paraphrased` | −0.0017, 4/10 — no advantage | +0.0392, 9/10 — unresolved (margin < Random's sd) |

### Verdict: the control splits the result in two

The control was decisive, and it cuts both ways. The earlier conclusion — that
Semantic-LRU's wins were "fully explained by not being recency-ordered" — turns
out to be **half right**.

**Where the embeddings contribute nothing (or hurt):**

- **`looping` is the headline correction.** This was Semantic-LRU's single
  largest win over LRU (+0.1992, 1.0 → 0.8008). Random eviction reaches
  **0.6875** — substantially *better* than Semantic-LRU, which loses to it on
  **9 of 10 seeds**. So not only is that win entirely explained by breaking
  recency, Semantic-LRU is a *worse* way of breaking it than a coin flip.
- **`random` trace**: all four policies are within noise of each other; Random
  is nominally best. Nothing to claim.

**Where the embeddings demonstrably do real work:**

- **`clustered`**: Semantic-LRU 0.4583 vs Random 0.5342 — a +0.0758 margin,
  better on **10/10 seeds**, exceeding Random's run-to-run spread. Note also
  that Random is clearly *worse* than LRU here (0.5342 vs 0.4725): on a trace
  with genuine locality, recency is valuable and randomness destroys it.
  Semantic-LRU roughly matches LRU while decisively beating Random — so the
  embeddings are recovering real signal, not just scrambling the order.
- **`sequential`**: Semantic-LRU 0.9400 vs Random 0.9858, better on 10/10 seeds.
- **`paraphrased`**: +0.0392 over Random on 9/10 seeds, but the margin is
  smaller than Random's own sd, so it is **suggestive and unresolved** at
  n=10 — not something to claim.

### The honest summary

1. **Semantic-LRU does not beat LRU on any trace at n=10.** Its two "robust"
   wins over LRU are on recency-pathological traces, and on the worst of those
   (`looping`) plain randomness beats it by a wide margin.
2. **But the embeddings are not inert.** On the two locality-bearing traces
   Semantic-LRU beats random eviction on 10/10 seeds, and on `clustered` it does
   so while random eviction is markedly worse than LRU. The similarity signal is
   real; it is simply not *better than recency* at deciding what to evict.
3. The defensible claim is therefore narrower than the original motivation:
   *Semantic-LRU makes meaningfully better-than-random eviction choices when the
   workload has semantic locality, but does not outperform LRU, and is a poor
   choice for breaking cyclic worst cases where random eviction does better.*

Remaining limitations worth stating: 20 pages, 5 resident, 120 accesses, one
corpus, n=10. `paraphrased` vs Random is unresolved at this sample size and
would need more seeds to settle.



Scheduler findings match the textbook: MLFQ gives the best response time
everywhere and the best waiting/turnaround on both non-uniform profiles, while
on `uniform` — where every job is the same size — preemption only adds overhead
and plain FCFS wins on waiting and turnaround.

---

## 3. Belady's Anomaly experiment

```bash
python -m benchmarks.belady_bench                      # 10 seeds, frames 2-10 (~2.5 min)
python -m benchmarks.belady_bench --seeds 5 --max-frames 12
```

Giving a page-replacement policy **more** memory should never make it worse.
For some policies that intuition is provably correct; for others it is simply
false. This experiment sweeps RAM capacity as an independent variable and looks
for capacity steps where adding a frame *increases* the fault rate.

### The theory being tested

| policy | stack algorithm? | prediction |
|---|---|---|
| **LRU** | **yes** — the pages resident in N frames are always a subset of those in N+1 | **provably immune.** An anomaly here would be an *implementation bug*, not a discovery — which is exactly why LRU is included: as a self-check on everything else. |
| **FIFO** | no | known to exhibit the anomaly; the canonical counterexample is reproduced below |
| **Semantic-LRU** | no — it evicts by embedding distance from the current query, not recency, so nothing guarantees the inclusion property | **genuinely open.** This is the question the experiment exists to answer about our own algorithm. |
| **Random** | no | no guarantee either way; serves as the control |

### 3a. Canonical reference string — run against the REAL kernel

The textbook string `1,2,3,4,1,2,5,1,2,3,4,5` at 3 vs 4 frames, replayed
through the actual `PageManager` (real page table, real eviction, real ChromaDB
swap). Memory starts **cold** — every page evicted to swap before the string
runs — because a reference string is defined over empty memory.

| policy | 3 frames | 4 frames | delta | anomaly? |
|---|---|---|---|---|
| **fifo** | 9 | **10** | **+1** | **YES** |
| lru | 10 | 8 | −2 | no |
| semantic_lru | 9 | 7 | −2 | no |
| random | 9 | 8 | −1 | no |

**FIFO 9→10 and LRU 10→8 match the published textbook values exactly.** That is
the headline result of this subsection, and it is a *validation of the kernel*,
not just of the benchmark: our FIFO and LRU reproduce the canonical fault counts
on the canonical input, so the replacement implementations behave as the
literature specifies.

### 3b. Capacity sweep — run against PolicySim, not the live kernel

> **Read this before quoting the sweep numbers.** The sweep does **not** run
> through the live kernel. It runs on `PolicySim`, an in-memory replica of the
> four replacement policies.

**Why.** Driving the sweep through the real `PageManager` costs roughly **8
seconds per cell**, essentially all of it ChromaDB persistent-write throughput
(~25 ops/sec). The sweep is 4 policies × 5 traces × 9 capacities × 10 seeds =
1800 cells — hours of vector-store I/O to measure something that is purely a
property of victim selection. `PolicySim` replays a reference string over the
same policies with no vector store at all, bringing the run to ~2.5 minutes.

**How it is cross-validated.** `validate_against_page_manager()` replays the
canonical string through both the replica and the real `PageManager` and
compares fault counts at both capacities. For **fifo, lru and semantic_lru the
counts match exactly**, and the benchmark prints that check every run — if they
ever diverge, the output says the sweep is unvalidated rather than quietly
reporting it anyway.

**`random` is excluded from that check**, and the reason matters: both
implementations use a seeded RNG, but they consume draws in a different order
(the real manager evicts inside a budget loop over its resident `OrderedDict`).
Real gives 9/8, the replica 10/6. That is a **draw-order difference, not a
semantic one** — the policy is the same in both, so a mismatch here is expected
and is not evidence that the replica is unfaithful.

**What the replica does *not* cover:** the semantic *lookup* path. In the real
kernel an access is a similarity search that resolves to some page; here an
access names its page directly. Belady's Anomaly is a property of the
replacement policy over a reference string, so this is the right model — but a
sweep result cannot speak to how query resolution interacts with capacity.

### 3c. Findings (10 seeds, frames 2–10, 5 traces, 4 policies)

**Zero anomalies detected.**

- **LRU — no anomaly at any capacity step, on any trace.** Consistent with the
  stack-algorithm proof. This is the self-check passing: had LRU shown one, the
  correct conclusion would have been "we have a bug", and the rest of the
  numbers would have been suspect.
- **FIFO — confirmed in principle, not triggered in practice.** It exhibits the
  anomaly on the canonical string (3a) but on none of our five traces. Our
  workloads simply do not contain the pathological interleaving. The honest
  statement is *"FIFO's anomaly is real and we reproduced it, but our own
  workloads never hit it"* — not "FIFO is fine here".
- **Semantic-LRU — a negative result, and NOT proof of immunity.** Across five
  traces and nine capacities it behaved monotonically: more frames never meant
  more faults. But Semantic-LRU has **no stack property**, so nothing forbids
  the anomaly; this is absence of evidence over one workload set, not evidence
  of absence. A different access pattern could still expose it, and claiming
  immunity would be claiming something we have not proved.
- **Random — no systematic anomaly** once the seed count was adequate (see the
  methodology note below, which is precisely about this policy).

### 3d. Methodology note: an anomaly that evaporated

The **5-seed** run reported one anomaly — Random policy on the `random` trace,
4→5 frames, +0.0267, holding on **3 of 5 seeds**. At **10 seeds it disappeared
entirely.** It was noise, and 3/5 is barely above chance.

Two things changed as a result, both worth stating because the first draft of
this experiment would have reported a finding that does not exist:

1. The `systematic` threshold was tightened from *">50% of seeds"* to
   **"≥75% of seeds"**. A step that clears 50% but not 75% is now labelled
   `weak` rather than promoted to a result.
2. Every reported anomaly carries `seeds_worse / seeds_total` so the strength of
   the evidence is visible in the output itself, not just in the mean.

### 3e. The detector was verified to actually fire

"No anomalies found" is only meaningful if the detector *can* find one. A
silently broken detector produces identical output. Two checks:

- `detect_anomalies()` on a synthetic rising curve (0.75 → 0.83 across ten
  seeds) returns exactly one anomaly, flagged `systematic`.
- `PolicySim` independently reproduces FIFO's canonical 9→10 on the reference
  string, so the machinery the sweep uses demonstrably *can* express an anomaly.

So the empty result reflects a working detector finding nothing, not a detector
finding nothing because it is broken.

### Artifacts

`belady.json` plus one chart per trace (`belady_<trace>.png`): mean fault rate
vs capacity, one line per policy, with any upward segment marked by a red arrow
and annotated with its magnitude and seed count. FIFO and LRU produce identical
curves on `clustered` and `paraphrased` — correct, since neither consults the
query and `paraphrased` reuses `clustered`'s access order — so the policies use
distinct dash patterns to keep a coincident line visible underneath another.
