"""Empirical comparison of the page-replacement policies (FIFO, LRU, Semantic-LRU).

This is the experiment that tests whether Semantic-LRU — the project's original
contribution — actually beats classical LRU, and on which access patterns.

Design
------
One fixed universe of pages (topic-clustered), exercised by four reproducible
access patterns, so the ONLY thing that varies between traces is the order of
access:

  sequential : 0,1,2,... straight through, repeatedly
  random     : uniformly random page ids (seeded)
  looping    : cycles over a working set larger than RAM — the classic
               pathological case for both LRU and FIFO (every access faults)
  clustered  : stays within one topic for a while, then switches topic —
               models an agent whose queries revisit related material

Each policy is run over each trace with a freshly built PageManager (its own
ChromaDB directory), primed by writing every page once, after which only the
trace's reads are measured.

An access is driven by querying with a page's own content, so the intended page
is the exact-match nearest neighbour; whether it is served from RAM (hit) or has
to be fetched from swap (page fault) is what we measure.

Embeddings: uses the real Ollama embedder so Semantic-LRU is evaluated fairly.
If Ollama is unavailable the run falls back to the hashing embedder and SAYS SO
in the output — Semantic-LRU results under hashing are not a fair test of it.
Embeddings are memoized per unique text; that is a pure speed optimization and
cannot change any result, since embedding is deterministic.

No tuning is applied to favour any policy. Results are reported as measured.
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from typing import Callable, Dict, List, Optional, Sequence

from kernel.memory import PageManager
from kernel.memory.embeddings import (
    Embedder,
    HashingEmbedder,
    OllamaEmbedder,
    build_embedder,
    set_embedder,
)

# --- fixed experiment parameters (reproducible + citable) -----------------
SEED = 20260726
DEFAULT_NUM_SEEDS = 10
TOKENS_PER_PAGE = 10
RAM_BUDGET_TOKENS = 50           # => 5 pages resident at once
PAGES_PER_TOPIC = 5
ACCESSES = 120
LOOP_WORKING_SET = 8             # > 5 resident pages, so looping thrashes
AGENT = "bench-agent"

# "random" is an EXPERIMENTAL CONTROL that exists only in this benchmark (see
# BenchPageManager); it is deliberately not a production policy in
# kernel/memory/replacement.py.
POLICIES = ("fifo", "lru", "semantic_lru", "random")
RANDOM_POLICY = "random"

# Four topics x 5 pages. Sentences within a topic are semantically related but
# deliberately varied in wording, so similarity has to come from meaning.
TOPICS: Dict[str, List[str]] = {
    "photosynthesis": [
        "Green plants convert sunlight into chemical energy stored as glucose.",
        "Chlorophyll in leaves absorbs light to drive carbon fixation.",
        "Carbon dioxide and water become sugar and oxygen inside chloroplasts.",
        "Foliage captures solar radiation to manufacture its own nourishment.",
        "The Calvin cycle assembles carbohydrate from atmospheric carbon.",
    ],
    "cpu_scheduling": [
        "Round robin assigns each process a fixed time quantum before preemption.",
        "A scheduler decides which task occupies the processor next.",
        "Shortest job first minimises average waiting time for batch work.",
        "Context switching saves and restores register state between tasks.",
        "Multilevel feedback queues demote jobs that consume long bursts.",
    ],
    "ocean_tides": [
        "Tides arise from gravitational attraction between the moon and earth.",
        "Coastal water levels rise and fall twice each lunar day.",
        "Spring tides occur when the sun and moon align their pull.",
        "The gravitational gradient across the planet deforms the sea surface.",
        "Neap tides happen when solar and lunar forces oppose each other.",
    ],
    "bread_baking": [
        "Yeast ferments dough and releases carbon dioxide that makes it rise.",
        "Kneading develops gluten strands that give loaves their structure.",
        "A hot oven sets the crust while steam keeps the crumb tender.",
        "Sourdough starters culture wild microbes for a tangy flavour.",
        "Proving the dough lets it relax and expand before baking.",
    ],
}

# HAND-WRITTEN paraphrase queries, one per page, fixed here so the trace is
# reproducible. Each asks for its page's idea using different words: content
# words are deliberately disjoint from the page text, so a lexical matcher has
# almost nothing to latch onto and only meaning can bridge the gap.
# `report_paraphrase_overlap()` measures the residual overlap rather than
# asserting it, and the hashing embedder's near-zero scores on these queries
# (see the paraphrase_lexical_baseline block in the output) demonstrate that
# word overlap alone cannot resolve them.
PARAPHRASE_QUERIES: Dict[str, str] = {
    "photosynthesis-0": "how vegetation makes fuel from daylight",
    "photosynthesis-1": "what green pigment captures illumination for sugar building",
    "photosynthesis-2": "gas plus liquid transformed to sweetness within plant organelles",
    "photosynthesis-3": "leaves harvest starlight energy making their food",
    "photosynthesis-4": "dark reactions build sugars using air borne gas",
    "cpu_scheduling-0": "cyclic dispatch giving every task an equal slice",
    "cpu_scheduling-1": "selecting which job runs on hardware first",
    "cpu_scheduling-2": "prioritising brief tasks to reduce queue delay",
    "cpu_scheduling-3": "swapping cpu contents when changing programs",
    "cpu_scheduling-4": "tiered priority dropping for compute hungry work",
    "ocean_tides-0": "why sea levels change due to lunar pull",
    "ocean_tides-1": "shoreline height fluctuating two times daily",
    "ocean_tides-2": "extreme high water during solar lunar alignment",
    "ocean_tides-3": "how mass attraction bulges ocean shape",
    "ocean_tides-4": "weakest range when celestial bodies pull crosswise",
    "bread_baking-0": "microbes producing gas so batter expands",
    "bread_baking-1": "working flour mixture to build elastic network",
    "bread_baking-2": "high heat firming exterior moist interior",
    "bread_baking-3": "natural leaven giving sour taste",
    "bread_baking-4": "resting risen mixture prior to the oven",
}


class BenchPageManager(PageManager):
    """PageManager plus one extra eviction policy that exists only for this
    benchmark: `random`, which evicts a uniformly random resident page.

    Why it exists: Semantic-LRU's robust wins are on traces where recency is
    pathological (`looping`, `sequential`), where *any* policy that breaks the
    recency ordering should do well. Random eviction is the null hypothesis for
    "the embeddings are doing real work" — if Random matches Semantic-LRU there,
    the embeddings contribute nothing measurable and the win is fully explained
    by not being recency-ordered.

    It is intentionally NOT added to kernel/memory/replacement.py: it is a
    scientific control, not a policy anyone should run in production.

    The override hooks `_make_room`, which is the single place PageManager
    chooses victims (used by both writes and page-fault reloads), and delegates
    every other policy untouched to the real implementation.
    """

    def __init__(self, *args, rng: Optional[random.Random] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # seeded per run so random eviction is reproducible
        self.rng = rng if rng is not None else random.Random(0)

    def _make_room(
        self,
        agent_id: str,
        incoming_tokens: int,
        query_embedding: List[float],
        policy: Optional[str] = None,
    ) -> List[str]:
        effective_policy = policy or self.policy
        if effective_policy != RANDOM_POLICY:
            return super()._make_room(
                agent_id, incoming_tokens, query_embedding, policy=policy
            )

        evicted: List[str] = []
        while (
            self.ram_tokens(agent_id) + incoming_tokens > self.ram_budget_tokens
            and self.ram[agent_id]
        ):
            victim_id = self.rng.choice(list(self.ram[agent_id].keys()))
            self._evict(agent_id, victim_id)
            evicted.append(victim_id)
        return evicted


class CachingEmbedder(Embedder):
    """Memoizes an inner embedder. Embedding is deterministic, so caching
    changes only speed, never results."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self.semantic = inner.semantic
        self._cache: Dict[str, List[float]] = {}

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def is_available(self) -> bool:
        return self.inner.is_available()

    def embed(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self.inner.embed(text)
        return self._cache[text]


def build_pages(seed: Optional[int] = None) -> List[Dict[str, object]]:
    """The page universe: stable ids, topic membership, and content.

    With a seed, the page ORDER is shuffled. The page_id -> content ->
    paraphrase mapping is never disturbed (so the hand-written paraphrases stay
    valid), but shuffling varies the corpus in ways that genuinely matter:
    the priming order — and therefore which pages start resident in RAM — and
    which pages the order-driven `sequential` and `looping` traces walk.
    """
    pages: List[Dict[str, object]] = []
    for topic, sentences in TOPICS.items():
        for i, text in enumerate(sentences[:PAGES_PER_TOPIC]):
            pages.append(
                {"page_id": f"{topic}-{i}", "topic": topic, "content": text}
            )
    if seed is not None:
        random.Random(f"{seed}-corpus").shuffle(pages)
    return pages


# --- access-pattern generators (each seeded for reproducibility) ----------

def trace_sequential(pages: Sequence[dict], n: int, seed: int) -> List[int]:
    return [i % len(pages) for i in range(n)]


def trace_random(pages: Sequence[dict], n: int, seed: int) -> List[int]:
    rng = random.Random(f"{seed}-random")
    return [rng.randrange(len(pages)) for _ in range(n)]


def trace_looping(pages: Sequence[dict], n: int, seed: int) -> List[int]:
    working_set = min(LOOP_WORKING_SET, len(pages))
    return [i % working_set for i in range(n)]


def trace_clustered(pages: Sequence[dict], n: int, seed: int) -> List[int]:
    """Dwell inside one topic for a run of accesses, then switch topics."""
    rng = random.Random(f"{seed}-clustered")
    by_topic: Dict[str, List[int]] = {}
    for idx, page in enumerate(pages):
        by_topic.setdefault(page["topic"], []).append(idx)
    topics = list(by_topic)

    trace: List[int] = []
    while len(trace) < n:
        topic = rng.choice(topics)
        for _ in range(rng.randint(4, 8)):
            if len(trace) >= n:
                break
            trace.append(rng.choice(by_topic[topic]))
    return trace


def trace_paraphrased(pages: Sequence[dict], n: int, seed: int) -> List[int]:
    """Identical access ORDER to `clustered` — only the query wording differs
    (paraphrases instead of the page's own text). Holding the order fixed makes
    clustered vs paraphrased a clean A/B on a single variable: whether the query
    lexically matches the page."""
    return trace_clustered(pages, n, seed)


TRACES: Dict[str, Callable[[Sequence[dict], int, int], List[int]]] = {
    "sequential": trace_sequential,
    "random": trace_random,
    "looping": trace_looping,
    "clustered": trace_clustered,
    "paraphrased": trace_paraphrased,
}

# traces whose accesses are driven by hand-written paraphrases rather than the
# page's own text
PARAPHRASE_TRACES = {"paraphrased"}

TRACE_DESCRIPTIONS = {
    "sequential": "walks every page in order, repeatedly",
    "random": "uniformly random page ids (seeded)",
    "looping": f"cycles over {LOOP_WORKING_SET} pages > RAM capacity (classic LRU worst case)",
    "clustered": "dwells within one topic, then switches - semantic locality",
    "paraphrased": (
        "same access order as 'clustered', but queried by hand-written "
        "paraphrases with minimal word overlap - only meaning can match"
    ),
}


def release_chroma_clients() -> None:
    """Drop ChromaDB's process-global cache of live clients.

    Each PersistentClient stays registered (holding sqlite/HNSW handles) until
    released. A multi-seed sweep opens ~150 of them, and letting them all stay
    live is the most plausible cause of the intermittent
    "Error creating hnsw segment reader: Nothing found on disk" seen mid-sweep.
    Called after each run, once that run's client is finished with. Safe: it
    only releases clients, it does not touch files on disk.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001 — cleanup must never mask results
        pass


def run_trace_resilient(
    policy: str,
    pages: Sequence[dict],
    trace: Sequence[int],
    paraphrased: bool,
    chroma_dirs: List[str],
    seed: int = SEED,
    attempts: int = 3,
) -> Dict[str, float]:
    """run_trace with a retry on ChromaDB's intermittent internal failure.

    ChromaDB occasionally raises InternalError ("Error creating hnsw segment
    reader: Nothing found on disk") partway through a long sweep of sequential
    persistent clients. It originates inside ChromaDB's Rust binding, not in
    this benchmark, and could not be reproduced in isolation.

    Retrying on a brand-new directory is measurement-safe rather than a fudge:
    a run is fully deterministic given (seed, pages, trace, policy) — the store
    is primed from scratch every time and embeddings are cached — so a retry
    recomputes exactly the same numbers. Nothing is skipped or approximated; the
    last attempt is allowed to raise so a persistent failure is never silently
    swallowed.
    """
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        run_dir = tempfile.mkdtemp(prefix=f"bench-{policy}-")
        chroma_dirs.append(run_dir)
        try:
            return run_trace(
                policy, pages, trace,
                paraphrased=paraphrased, chroma_dir=run_dir, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 — retried below, re-raised if final
            last_error = exc
            if attempt == attempts - 1:
                raise
            print(
                f"    [retry {attempt + 1}/{attempts - 1}] {policy}: "
                f"{type(exc).__name__} - retrying on a fresh store",
                flush=True,
            )
            release_chroma_clients()
        finally:
            release_chroma_clients()
    raise last_error  # unreachable, keeps type checkers happy


def release_chroma_dirs(dirs: Sequence[str]) -> None:
    """Release every cached ChromaDB client, then remove its directory.

    Order matters: clearing the client cache first closes the sqlite/HNSW
    handles, so the directories can actually be removed (on Windows an open
    handle makes rmtree a silent no-op, which is how stale directories piled up
    and eventually broke the sweep)."""
    release_chroma_clients()
    for path in dirs:
        shutil.rmtree(path, ignore_errors=True)


def query_for(page: dict, paraphrased: bool) -> str:
    """The text used to access a page: its own content, or its paraphrase."""
    if not paraphrased:
        return page["content"]
    return PARAPHRASE_QUERIES[page["page_id"]]


def run_trace(
    policy: str,
    pages: Sequence[dict],
    trace: Sequence[int],
    paraphrased: bool = False,
    chroma_dir: Optional[str] = None,
    seed: int = SEED,
) -> Dict[str, float]:
    """Prime a fresh PageManager with every page, then replay the trace.

    Each run gets its own `chroma_dir`, and — importantly — that directory is
    NOT deleted here. ChromaDB keeps a per-path client alive in a process-global
    cache with sqlite/HNSW handles open; removing the files underneath a live
    client leaves a stale segment reader and later runs die with
    "Error creating hnsw segment reader: Nothing found on disk". (Deleting and
    recreating collections on a shared client fails the same way.) So the caller
    accumulates the directories and removes them all once the sweep is over,
    when every client is finished with.
    """
    owns_dir = chroma_dir is None
    chroma_dir = chroma_dir or tempfile.mkdtemp(prefix="bench-")
    try:
        manager = BenchPageManager(
            ram_budget_tokens=RAM_BUDGET_TOKENS,
            policy=policy,
            chroma_path=chroma_dir,
            # seeded per (seed, policy) so random eviction is reproducible
            rng=random.Random(f"{seed}-evict-{policy}"),
        )
        for page in pages:
            manager.write_page(
                AGENT, page["page_id"], page["content"], token_count=TOKENS_PER_PAGE
            )

        by_id = {p["page_id"]: p for p in pages}
        faults = 0
        correct = 0
        topic_hits = 0
        retrieved_ids: set = set()
        residency: List[int] = []
        for idx in trace:
            page = pages[idx]
            result = manager.read(AGENT, query_for(page, paraphrased))
            if result.page_fault:
                faults += 1
            if result.page.page_id == page["page_id"]:
                correct += 1
            retrieved = by_id.get(result.page.page_id)
            if retrieved is not None and retrieved["topic"] == page["topic"]:
                topic_hits += 1
            retrieved_ids.add(result.page.page_id)
            residency.append(len(manager.ram[AGENT]))

        n = len(trace)
        return {
            "accesses": n,
            "page_faults": faults,
            "page_fault_rate": round(faults / n, 4),
            "hit_ratio": round(1 - faults / n, 4),
            "avg_pages_in_ram": round(statistics.fmean(residency), 3),
            # fraction of reads returning the exact intended page. 1.0 for
            # exact-text traces; under paraphrase it measures how often meaning
            # alone resolved to the right page.
            "retrieval_accuracy": round(correct / n, 4),
            # looser: did we at least land in the right topic? A paraphrase that
            # retrieves a sibling page is a near-miss, not a total failure.
            "topic_match_rate": round(topic_hits / n, 4),
            # how many distinct pages the trace actually touched *after*
            # retrieval. Under paraphrase, several queries can resolve to the
            # same nearest page, shrinking the effective working set - which
            # lowers fault rates for reasons unrelated to the policy. Recorded
            # so cross-trace fault rates are not naively compared.
            "distinct_pages_retrieved": len(retrieved_ids),
        }
    finally:
        # Only tear down when this call created the directory itself (standalone
        # use). In a sweep the caller owns cleanup — see the docstring.
        if owns_dir:
            release_chroma_dirs([chroma_dir])


def measure_paraphrase_overlap(pages: Sequence[dict]) -> Dict[str, float]:
    """Quantify how lexically distinct the paraphrases are from their pages, and
    how well a purely lexical matcher (the hashing embedder) can resolve them.

    This substantiates the 'minimal word overlap' claim with numbers instead of
    asserting it, and shows the paraphrase trace really does require meaning:
    if the hashing baseline could rank the correct page first, the trace would
    not be testing semantics at all.
    """
    import re

    tokens = re.compile(r"[a-z0-9]+")
    lexical = HashingEmbedder()

    jaccards: List[float] = []
    shared_words = 0
    lexical_correct = 0
    for page in pages:
        query = PARAPHRASE_QUERIES[page["page_id"]]
        a = set(tokens.findall(page["content"].lower()))
        b = set(tokens.findall(query.lower()))
        jaccards.append(len(a & b) / len(a | b) if a | b else 0.0)
        shared_words += len(a & b)

        # nearest page under pure lexical similarity
        qv = lexical.embed(query)
        best_id, best_score = None, -1.0
        for candidate in pages:
            cv = lexical.embed(candidate["content"])
            score = sum(x * y for x, y in zip(qv, cv))
            if score > best_score:
                best_id, best_score = candidate["page_id"], score
        if best_id == page["page_id"]:
            lexical_correct += 1

    n = len(pages)
    return {
        "mean_jaccard_overlap": round(statistics.fmean(jaccards), 4),
        "max_jaccard_overlap": round(max(jaccards), 4),
        "total_shared_words": shared_words,
        "mean_shared_words_per_pair": round(shared_words / n, 3),
        "lexical_top1_accuracy": round(lexical_correct / n, 4),
    }


def select_embedder() -> Embedder:
    """Prefer the real Ollama embedder; fall back to hashing, loudly."""
    candidate = build_embedder()
    if not isinstance(candidate, OllamaEmbedder):
        candidate = HashingEmbedder()
    return CachingEmbedder(candidate)


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Mean and sample standard deviation (n-1); std is 0.0 for a single value."""
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def compare_paired(
    per_seed: Dict[str, List[float]], baseline: str, challenger: str
) -> Dict:
    """Paired comparison of two policies across seeds on `page_fault_rate`.

    Every policy sees the identical corpus and access order within a seed, so
    seed-to-seed variance is common-mode and cancels when differences are taken
    per seed. That makes the paired spread a far sharper test than comparing two
    independent mean +/- std summaries, whose error bars can overlap heavily even
    when one policy wins on every single seed.
    """
    diffs = [b - c for b, c in zip(per_seed[baseline], per_seed[challenger])]
    wins = sum(1 for d in diffs if d > 0)     # challenger had fewer faults
    losses = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)

    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    baseline_std = statistics.stdev(per_seed[baseline]) if len(diffs) > 1 else 0.0
    margin = statistics.fmean(per_seed[baseline]) - statistics.fmean(per_seed[challenger])

    return {
        "baseline": baseline,
        "challenger": challenger,
        "metric": "page_fault_rate",
        "mean_improvement": round(mean_diff, 4),
        "std_of_paired_differences": round(std_diff, 4),
        "seeds_challenger_better": wins,
        "seeds_challenger_worse": losses,
        "seeds_tied": ties,
        # the headline honesty check requested: is the gap between the two means
        # bigger than the run-to-run spread of the baseline?
        "margin_between_means": round(margin, 4),
        "baseline_std": round(baseline_std, 4),
        "margin_exceeds_baseline_std": bool(abs(margin) > baseline_std),
        # the paired equivalent: is the mean improvement bigger than its own spread?
        "improvement_exceeds_paired_std": bool(abs(mean_diff) > std_diff),
    }


def run_benchmark(num_seeds: int = DEFAULT_NUM_SEEDS) -> Dict:
    embedder = select_embedder()
    set_embedder(embedder)
    # every run gets its own directory; all are removed together at the end,
    # once no client is still holding them open (see release_chroma_dirs)
    chroma_dirs: List[str] = []
    try:
        seeds = [SEED + i for i in range(num_seeds)]
        # the lexical baseline is a property of the fixed corpus text, not of
        # any seed, so it is measured once on the unshuffled corpus
        reference_pages = build_pages()

        results: Dict = {
            "benchmark": "memory",
            "parameters": {
                "base_seed": SEED,
                "num_seeds": num_seeds,
                "seeds": seeds,
                "num_pages": len(reference_pages),
                "topics": list(TOPICS),
                "pages_per_topic": PAGES_PER_TOPIC,
                "tokens_per_page": TOKENS_PER_PAGE,
                "ram_budget_tokens": RAM_BUDGET_TOKENS,
                "pages_resident_at_once": RAM_BUDGET_TOKENS // TOKENS_PER_PAGE,
                "accesses_per_trace": ACCESSES,
                "loop_working_set": LOOP_WORKING_SET,
                "embedder": embedder.describe(),
                "embeddings_are_semantic": embedder.semantic,
                "seed_varies": (
                    "page/priming order (corpus) and access ordering; the "
                    "hand-written paraphrase queries are fixed by construction"
                ),
            },
            "paraphrase_lexical_baseline": measure_paraphrase_overlap(reference_pages),
            "traces": {},
        }

        for trace_name, generator in TRACES.items():
            paraphrased = trace_name in PARAPHRASE_TRACES
            per_seed: Dict[str, Dict[str, List[float]]] = {
                policy: {} for policy in POLICIES
            }
            unique_touched: List[int] = []

            for seed in seeds:
                pages = build_pages(seed)
                trace = generator(pages, ACCESSES, seed)
                unique_touched.append(len(set(trace)))
                for policy in POLICIES:
                    run = run_trace_resilient(
                        policy, pages, trace, paraphrased, chroma_dirs, seed=seed
                    )
                    for metric, value in run.items():
                        per_seed[policy].setdefault(metric, []).append(value)

            trace_result: Dict = {
                "description": TRACE_DESCRIPTIONS[trace_name],
                "queries": "paraphrase" if paraphrased else "exact page text",
                "unique_pages_touched_mean": round(
                    statistics.fmean(unique_touched), 2
                ),
                "policies": {},
            }
            for policy in POLICIES:
                metrics = per_seed[policy]
                summary: Dict = {
                    "page_fault_rate": _mean_std(metrics["page_fault_rate"]),
                    "hit_ratio": _mean_std(metrics["hit_ratio"]),
                    "per_seed_page_fault_rate": metrics["page_fault_rate"],
                }
                # remaining metrics summarised by mean only
                for metric in (
                    "avg_pages_in_ram",
                    "retrieval_accuracy",
                    "topic_match_rate",
                    "distinct_pages_retrieved",
                    "page_faults",
                ):
                    summary[f"{metric}_mean"] = round(
                        statistics.fmean(metrics[metric]), 4
                    )
                trace_result["policies"][policy] = summary

            fault_rates = {
                policy: per_seed[policy]["page_fault_rate"] for policy in POLICIES
            }
            trace_result["semantic_vs_lru"] = compare_paired(
                fault_rates, baseline="lru", challenger="semantic_lru"
            )
            # the control: does Semantic-LRU beat a policy that merely breaks
            # recency ordering? If not, the embeddings add nothing measurable.
            trace_result["semantic_vs_random"] = compare_paired(
                fault_rates, baseline=RANDOM_POLICY, challenger="semantic_lru"
            )
            results["traces"][trace_name] = trace_result

        return results
    finally:
        set_embedder(None)
        release_chroma_dirs(chroma_dirs)


METRIC_LABELS = {
    "page_fault_rate": "fault rate",
    "hit_ratio": "hit ratio",
    "avg_pages_in_ram": "avg pages RAM",
    "page_faults": "faults",
    "retrieval_accuracy": "retr. acc",
    "topic_match_rate": "topic match",
    "distinct_pages_retrieved": "distinct pages",
}


def verdict_for(cmp: Dict, n_seeds: int) -> str:
    """A deliberately conservative one-line verdict.

    Two different questions get two different answers and both are reported,
    because quoting only one would mislead:
      * unpaired - is the gap between the means bigger than the run-to-run
        spread of the baseline? (the strict bar the brief asked about)
      * paired   - is the per-seed improvement consistent in sign and larger
        than its own spread? (sharper, since seeds are shared)
    A margin below the unpaired spread is NEVER reported as a plain "win".
    """
    margin = cmp["margin_between_means"]
    if margin < 0:
        return (
            f"NO ADVANTAGE - Semantic-LRU is worse on average "
            f"(better on only {cmp['seeds_challenger_better']}/{n_seeds} seeds)"
        )
    if margin == 0:
        return "NO DIFFERENCE - identical mean fault rate"

    baseline = cmp["baseline"]
    wins = f"{cmp['seeds_challenger_better']}/{n_seeds} seeds"
    consistent = (
        cmp["improvement_exceeds_paired_std"] and cmp["seeds_challenger_worse"] == 0
    )
    if cmp["margin_exceeds_baseline_std"]:
        return (
            f"ROBUST - margin exceeds {baseline}'s run-to-run spread; wins on {wins}"
            + (", every seed" if consistent else "")
        )
    if cmp["improvement_exceeds_paired_std"]:
        return (
            f"UNRESOLVED - wins on {wins} and the paired improvement exceeds its "
            f"own spread, but the margin is still smaller than {baseline}'s "
            "run-to-run spread; suggestive, not established at this sample size"
        )
    return (
        f"WITHIN NOISE - margin is smaller than {baseline}'s run-to-run spread "
        f"and the paired improvement is smaller than its own spread (wins on "
        f"{wins}); not a demonstrated win"
    )


def format_tables(results: Dict) -> str:
    lines: List[str] = []
    p = results["parameters"]
    lines.append("=" * 78)
    lines.append("MEMORY / PAGE-REPLACEMENT BENCHMARK")
    lines.append("=" * 78)
    lines.append(
        f"base_seed={p['base_seed']}  seeds={p['num_seeds']}  "
        f"pages={p['num_pages']} ({len(p['topics'])} topics x "
        f"{p['pages_per_topic']})  ram={p['ram_budget_tokens']} tokens "
        f"(~{p['pages_resident_at_once']} resident)  accesses={p['accesses_per_trace']}"
    )
    lines.append(f"per-seed variation: {p['seed_varies']}")
    lines.append(f"embedder: {p['embedder']}")
    if not p["embeddings_are_semantic"]:
        lines.append(
            "  !! WARNING: hashing embeddings in use - Semantic-LRU is NOT being"
        )
        lines.append(
            "     fairly evaluated here. Start Ollama + `ollama pull nomic-embed-text`."
        )

    base = results.get("paraphrase_lexical_baseline")
    if base:
        lines.append("")
        lines.append("paraphrase queries (hand-written) vs their page text:")
        lines.append(
            f"    mean Jaccard overlap={base['mean_jaccard_overlap']:g} "
            f"(max {base['max_jaccard_overlap']:g}), "
            f"{base['mean_shared_words_per_pair']:g} shared words per pair"
        )
        lines.append(
            f"    lexical-only top-1 accuracy={base['lexical_top1_accuracy']:g} "
            f"-> word overlap alone {'CANNOT' if base['lexical_top1_accuracy'] < 0.5 else 'can'} "
            f"resolve these queries"
        )

    for trace_name, data in results["traces"].items():
        lines.append("")
        lines.append(f"--- trace: {trace_name} ---")
        lines.append(f"    {data['description']}")
        lines.append(
            f"    mean unique pages touched: {data['unique_pages_touched_mean']}"
            f"   |  queries: {data.get('queries', 'exact page text')}"
        )
        lines.append("")

        header = [
            "policy", "fault rate (mean+/-sd)", "hit ratio (mean+/-sd)",
            "range", "retr. acc", "topic match",
        ]
        rows = []
        for pol in POLICIES:
            d = data["policies"][pol]
            fr, hr = d["page_fault_rate"], d["hit_ratio"]
            rows.append([
                pol,
                f"{fr['mean']:.4f} +/- {fr['std']:.4f}",
                f"{hr['mean']:.4f} +/- {hr['std']:.4f}",
                f"{fr['min']:.3f}-{fr['max']:.3f}",
                f"{d['retrieval_accuracy_mean']:g}",
                f"{d['topic_match_rate_mean']:g}",
            ])
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

        def fmt(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        lines.append("    " + fmt(header))
        lines.append("    " + "  ".join("-" * x for x in widths))
        for row in rows:
            lines.append("    " + fmt(row))

        cmp = data["semantic_vs_lru"]
        n_seeds = results["parameters"]["num_seeds"]
        lines.append("")
        lines.append(
            f"    Semantic-LRU vs LRU:  mean margin={cmp['margin_between_means']:+.4f}  "
            f"(LRU run-to-run sd={cmp['baseline_std']:.4f})"
        )
        lines.append(
            f"      paired: improvement={cmp['mean_improvement']:+.4f} "
            f"+/- {cmp['std_of_paired_differences']:.4f}   "
            f"better on {cmp['seeds_challenger_better']}/{n_seeds} seeds, "
            f"worse on {cmp['seeds_challenger_worse']}, tied on {cmp['seeds_tied']}"
        )

        lines.append(f"      verdict: {verdict_for(cmp, n_seeds)}")

        ctrl = data.get("semantic_vs_random")
        if ctrl:
            lines.append("")
            lines.append(
                f"    CONTROL - Semantic-LRU vs Random eviction:  "
                f"mean margin={ctrl['margin_between_means']:+.4f}  "
                f"(Random run-to-run sd={ctrl['baseline_std']:.4f})"
            )
            lines.append(
                f"      paired: improvement={ctrl['mean_improvement']:+.4f} "
                f"+/- {ctrl['std_of_paired_differences']:.4f}   "
                f"better on {ctrl['seeds_challenger_better']}/{n_seeds} seeds, "
                f"worse on {ctrl['seeds_challenger_worse']}, tied on {ctrl['seeds_tied']}"
            )
            lines.append(f"      verdict: {verdict_for(ctrl, n_seeds)}")
            if ctrl["margin_between_means"] <= 0:
                lines.append(
                    "      => embeddings provide NO benefit here: random eviction "
                    "matches or beats Semantic-LRU"
                )
            elif ctrl["margin_exceeds_baseline_std"]:
                lines.append(
                    "      => embeddings DO real work: Semantic-LRU beats mere "
                    "recency-breaking"
                )
            else:
                lines.append(
                    "      => embeddings may help, but the gap is not resolved at "
                    "this sample size"
                )

    return "\n".join(lines)


def main(num_seeds: int = DEFAULT_NUM_SEEDS) -> Dict:
    results = run_benchmark(num_seeds=num_seeds)
    print(format_tables(results))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="page-replacement benchmark")
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"number of seeds to average over (default {DEFAULT_NUM_SEEDS})",
    )
    main(num_seeds=parser.parse_args().seeds)
