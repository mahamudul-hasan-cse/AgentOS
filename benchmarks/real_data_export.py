"""Extract reusable benchmark workloads from a *real* syscall log.

The professor's requirement is that scheduler and memory benches can run on
captured execution, not only on seeded synthetic generators. This script is
the bridge:

  * read a live dispatcher's syscall log (and replay snapshots, when present,
    for process priority);
  * optionally *produce* that log by running the flagship pipeline and the
    in-kernel assistant against real providers;
  * write a JSON workload the existing harnesses can replay.

Scheduler jobs
--------------
Each completed LLM_CALL / TOOL_CALL becomes one process:
  arrival_time = wall-clock timestamp of the syscall, shifted so the first
                 event is t=0 (seconds);
  burst        = measured `latency_ms` / 1000 (the real provider/sandbox time);
  priority     = the agent's registered scheduler priority if a SPAWN_AGENT
                 record or a replay snapshot recorded one, else a documented
                 default (coordinator=0, everyone else=1).

Spawn-time `estimated_burst` is *not* used as the burst — that number is an
a-priori guess (2.0 / 4.0 in the pipeline), not the measured latency.

Memory accesses
---------------
MEM_WRITE / FILE_WRITE contribute pages (real content, real ids).
MEM_READ / FILE_SEARCH / FILE_READ contribute reads (real query text, real
order). FILE_* are included because the live pipeline and assistant store
working state in the semantic filesystem, not only in page memory; the page-
replacement harness replays that same content/query sequence.

The capture harness also writes each produced artifact into page memory via
MEM_WRITE and re-queries with the session's own FILE_SEARCH texts, so those
MEM_* records are themselves part of the live run rather than invented later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kernel.access_control import AccessControl, AgentPrivilege, QuotaManager
from kernel.filesystem import SemanticFS
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType

FORMAT = "agentos-real-workload/v1"
LEGACY_FORMAT = "aios-real-workload/v1"
WORKLOAD_FORMATS = frozenset({FORMAT, LEGACY_FORMAT})
DEFAULT_WORKLOAD_PATH = Path(__file__).resolve().parent / "workloads" / "real_captured.json"
DEFAULT_CONCURRENT_WORKLOAD_PATH = (
    Path(__file__).resolve().parent / "workloads" / "real_captured_concurrent.json"
)

SCHEDULER_TYPES = frozenset({"LLM_CALL", "TOOL_CALL"})
MEMORY_WRITE_TYPES = frozenset({"MEM_WRITE", "FILE_WRITE"})
MEMORY_READ_TYPES = frozenset({"MEM_READ", "FILE_SEARCH", "FILE_READ"})

CAPTURE_QUOTAS = {
    "coordinator": {"max_pages": 64, "max_calls_per_minute": 30},
    "researcher": {"max_pages": 64, "max_calls_per_minute": 30},
    "coder": {"max_pages": 64, "max_calls_per_minute": 30},
    "tester": {"max_pages": 64, "max_calls_per_minute": 30},
    "writer": {"max_pages": 64, "max_calls_per_minute": 30},
}

DEFAULT_PIPELINE_TASKS = (
    "Write a Python function that returns the nth Fibonacci number iteratively "
    "and print fib(10).",
    "Write a Python script that counts word frequencies in the string "
    "'the cat sat on the mat' and prints the counts sorted by word.",
    "Write a Python function that checks whether a string is a palindrome "
    "and demonstrate it on 'racecar'.",
)

DEFAULT_ASSISTANT_QUESTIONS = (
    "What processes are currently running?",
    "How does paging work in this kernel?",
    "What did the starvation benchmark find?",
    "How are provider rate-limit pools allocated?",
    "What syscalls have you issued recently?",
)


async def _set_quota_via_syscall(
    dispatcher: SyscallDispatcher,
    agent_id: str,
    *,
    max_pages: int = 200,
    max_calls_per_minute: int = 40,
) -> None:
    """Raise capture quotas through SET_QUOTA (logged), not a direct manager call."""
    dispatcher.acl.registry.register(dispatcher.KERNEL_AGENT, AgentPrivilege.KERNEL)
    await dispatcher.dispatch(
        dispatcher.KERNEL_AGENT,
        SyscallType.SET_QUOTA,
        target_agent_id=agent_id,
        max_pages=max_pages,
        max_calls_per_minute=max_calls_per_minute,
    )


def _enum(value: Any) -> str:
    return getattr(value, "value", value) if value is not None else ""


def as_record(entry: Any) -> Dict[str, Any]:
    """Normalise a Syscall dataclass or an already-serialised dict."""
    if hasattr(entry, "as_dict"):
        return entry.as_dict()
    if isinstance(entry, dict):
        return entry
    raise TypeError(f"cannot read syscall record of type {type(entry)!r}")


def _default_priority(agent_id: str) -> int:
    if agent_id.endswith("_coordinator") or agent_id == "kernel":
        return 0
    return 1


def process_index_from_syscalls(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """pid -> {priority, estimated_burst} from SPAWN_AGENT args/results."""
    index: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if _enum(rec.get("type")) != "SPAWN_AGENT":
            continue
        args = rec.get("args") or {}
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        pid = args.get("pid") or result.get("pid")
        if not pid:
            continue
        priority = result.get("priority", args.get("priority"))
        if priority is None:
            continue
        index[pid] = {
            "priority": int(priority),
            "estimated_burst": result.get("estimated_burst", args.get("estimated_burst")),
        }
    return index


def process_index_from_snapshots(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """pid -> {priority, ...} from replay snapshots (more complete mid-run)."""
    index: Dict[str, Dict[str, Any]] = {}
    for snap in snapshots:
        for proc in snap.get("processes") or []:
            pid = proc.get("pid")
            if not pid:
                continue
            entry = dict(index.get(pid, {}))
            if proc.get("priority") is not None:
                entry["priority"] = int(proc["priority"])
            if proc.get("estimated_burst") is not None:
                entry["estimated_burst"] = proc["estimated_burst"]
            index[pid] = entry
    return index


def extract_scheduler_processes(
    records: Sequence[Dict[str, Any]],
    process_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One scheduling job per measured LLM_CALL / TOOL_CALL."""
    index = dict(process_index or {})
    calls = [
        rec
        for rec in records
        if _enum(rec.get("type")) in SCHEDULER_TYPES and rec.get("latency_ms") is not None
    ]
    if not calls:
        return []

    t0 = min(float(rec["timestamp"]) for rec in calls)
    per_agent: Dict[str, int] = {}
    processes: List[Dict[str, Any]] = []
    for rec in calls:
        agent = rec["agent_id"]
        per_agent[agent] = per_agent.get(agent, 0) + 1
        meta = index.get(agent, {})
        burst = max(float(rec["latency_ms"]) / 1000.0, 1e-6)
        processes.append(
            {
                "pid": f"{agent}#{per_agent[agent]:02d}",
                "agent_id": agent,
                "arrival_time": round(float(rec["timestamp"]) - t0, 6),
                "burst": round(burst, 6),
                "priority": int(meta["priority"]) if meta.get("priority") is not None else _default_priority(agent),
                "syscall_type": _enum(rec.get("type")),
                "syscall_id": rec.get("syscall_id"),
                "status": _enum(rec.get("status")),
            }
        )
    return processes


def overlap_stats(processes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Did any job arrive while another job's measured burst was still open?

    The scheduler replay treats each LLM_CALL/TOOL_CALL as a CPU burst on a
    single processor. A ready queue only forms if intervals
    [arrival, arrival+burst) overlap — i.e. genuine concurrency in the log,
    not merely many jobs listed in order.
    """
    jobs = [
        {
            "pid": p["pid"],
            "start": float(p["arrival_time"]),
            "end": float(p["arrival_time"]) + float(p["burst"]),
        }
        for p in processes
    ]
    if not jobs:
        return {
            "jobs": 0,
            "arrivals_while_another_in_flight": 0,
            "max_concurrent_intervals": 0,
            "ready_queue_forms": False,
        }

    overlapping_arrivals = 0
    max_concurrent = 1
    for job in jobs:
        in_flight = [
            other
            for other in jobs
            if other["pid"] != job["pid"]
            and other["start"] <= job["start"] < other["end"]
        ]
        if in_flight:
            overlapping_arrivals += 1
        max_concurrent = max(max_concurrent, 1 + len(in_flight))

        mid = (job["start"] + job["end"]) / 2.0
        concurrent_at_mid = sum(
            1 for other in jobs if other["start"] <= mid < other["end"]
        )
        max_concurrent = max(max_concurrent, concurrent_at_mid)

    return {
        "jobs": len(jobs),
        "arrivals_while_another_in_flight": overlapping_arrivals,
        "max_concurrent_intervals": max_concurrent,
        "ready_queue_forms": bool(overlapping_arrivals > 0 or max_concurrent >= 2),
    }


def _topic_from_page_id(page_id: str, agent_id: str) -> str:
    if "__" in page_id:
        return page_id.split("__", 1)[0]
    if page_id.startswith("pipeline_") or "pipeline_" in agent_id:
        return "pipeline"
    if agent_id == "assistant":
        return "assistant"
    return agent_id


def _write_page_id(rec: Dict[str, Any]) -> Optional[str]:
    args = rec.get("args") or {}
    result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
    stype = _enum(rec.get("type"))
    if stype == "MEM_WRITE":
        return args.get("page_id") or (result.get("page") or {}).get("page_id")
    if stype == "FILE_WRITE":
        return args.get("filename") or result.get("filename")
    return None


def _write_content(rec: Dict[str, Any]) -> Optional[str]:
    args = rec.get("args") or {}
    result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
    stype = _enum(rec.get("type"))
    if stype == "MEM_WRITE":
        content = args.get("content")
        if content:
            return str(content)
        page = result.get("page") or {}
        if page.get("content"):
            return str(page["content"])
        return None
    if stype == "FILE_WRITE":
        content = args.get("content") or result.get("content")
        return str(content) if content else None
    return None


def _read_query(rec: Dict[str, Any]) -> Optional[str]:
    args = rec.get("args") or {}
    result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
    stype = _enum(rec.get("type"))
    if stype == "MEM_READ":
        return args.get("query_text")
    if stype == "FILE_SEARCH":
        return args.get("query") or result.get("query")
    if stype == "FILE_READ":
        return args.get("filename") or result.get("filename")
    return None


def _read_intended_page(rec: Dict[str, Any]) -> Optional[str]:
    result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
    args = rec.get("args") or {}
    stype = _enum(rec.get("type"))
    if stype == "MEM_READ":
        return (result.get("page") or {}).get("page_id")
    if stype == "FILE_SEARCH":
        matches = result.get("results") or []
        if matches and isinstance(matches[0], dict):
            return matches[0].get("filename")
        return None
    if stype == "FILE_READ":
        return args.get("filename") or result.get("filename")
    return None


def extract_memory_trace(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Real page universe + access sequence, in original syscall order."""
    pages: Dict[str, Dict[str, Any]] = {}
    accesses: List[Dict[str, Any]] = []

    for rec in records:
        stype = _enum(rec.get("type"))
        agent = rec.get("agent_id") or "unknown"
        if stype in MEMORY_WRITE_TYPES:
            page_id = _write_page_id(rec)
            content = _write_content(rec)
            if not page_id or content is None:
                continue
            pages[page_id] = {
                "page_id": page_id,
                "content": content,
                "topic": _topic_from_page_id(page_id, agent),
            }
            accesses.append(
                {
                    "op": "write",
                    "page_id": page_id,
                    "content": content,
                    "query": None,
                    "agent_id": agent,
                    "syscall_type": stype,
                }
            )
        elif stype in MEMORY_READ_TYPES:
            query = _read_query(rec)
            if not query:
                continue
            intended = _read_intended_page(rec)
            if stype == "FILE_READ" and intended:
                result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
                content = result.get("content")
                if content and intended not in pages:
                    pages[intended] = {
                        "page_id": intended,
                        "content": str(content),
                        "topic": _topic_from_page_id(intended, agent),
                    }
            accesses.append(
                {
                    "op": "read",
                    "page_id": intended,
                    "content": None,
                    "query": query,
                    "agent_id": agent,
                    "syscall_type": stype,
                }
            )

    return {"pages": list(pages.values()), "accesses": accesses}


def build_workload(
    syscalls: Sequence[Any],
    snapshots: Optional[Sequence[Dict[str, Any]]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    records = [as_record(s) for s in syscalls]
    index = process_index_from_syscalls(records)
    if snapshots:
        index.update(process_index_from_snapshots(list(snapshots)))

    scheduler_procs = extract_scheduler_processes(records, index)
    memory = extract_memory_trace(records)
    contention = overlap_stats(scheduler_procs)
    type_counts: Dict[str, int] = {}
    for rec in records:
        key = _enum(rec.get("type")) or "?"
        type_counts[key] = type_counts.get(key, 0) + 1

    return {
        "format": FORMAT,
        "source": "real",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "syscall_count": len(records),
            "syscall_type_counts": dict(sorted(type_counts.items())),
            "note": (
                "Scheduler bursts are measured LLM_CALL/TOOL_CALL latencies "
                "(seconds). Memory pages and accesses are the real "
                "MEM_READ/MEM_WRITE/FILE_SEARCH (plus FILE_WRITE/FILE_READ) "
                "sequence from this session, in original order, with original "
                "content. Replay snapshots contribute process priority only; "
                "they do not store page content."
            ),
            **(provenance or {}),
        },
        "scheduler": {
            "processes": scheduler_procs,
            "contention": contention,
        },
        "memory": memory,
    }


def load_syscall_dump(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Accept a raw list, `{syscalls, snapshots}`, or a pipeline result."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [as_record(s) for s in data], []
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a syscall dump")
    if data.get("format") in WORKLOAD_FORMATS:
        raise ValueError(
            f"{path} is already an exported workload; pass it to the benches "
            "with --workload-source real, not back through the exporter"
        )
    syscalls = data.get("syscalls") or data.get("log") or []
    snapshots = data.get("snapshots") or []
    return [as_record(s) for s in syscalls], list(snapshots)


def load_workload(path: Optional[Path] = None) -> Dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_WORKLOAD_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"real workload file not found: {target}. "
            "Run `python -m benchmarks.real_data_export --capture` first."
        )
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("source") != "real":
        raise ValueError(f"{target} is not a real-data workload file")
    return data


def write_workload(workload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload, indent=2), encoding="utf-8")
    return path


def _snapshots_from_dispatcher(dispatcher: SyscallDispatcher) -> List[Dict[str, Any]]:
    recorder = getattr(dispatcher, "recorder", None)
    if recorder is None:
        return []
    return [s.as_dict() for s in recorder.snapshots]


def _choose_driver() -> str:
    from kernel.drivers import DRIVER_REGISTRY

    # Prefer a provider that is actually reachable. Groq's configured
    # llama-3.1-8b-instant was retired (404); a local Ollama is the reliable
    # source of measured LLM latency on this machine.
    for name in ("ollama", "gemini", "groq"):
        cls = DRIVER_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            if cls().is_available():
                return name
        except Exception:  # noqa: BLE001 — try the next provider
            continue
    return "groq"


async def _persist_session_memory(dispatcher: SyscallDispatcher) -> Dict[str, int]:
    """Write this session's real artifacts into page memory and re-query them.

    The pipeline/assistant already issued FILE_* and LLM_CALL. This pass issues
    MEM_WRITE of those *same* contents and MEM_READ of the session's own
    FILE_SEARCH queries, so the log contains genuine MEM_* records whose
    content and query text came from the live run — not a synthetic pattern.
    """
    writes: List[Tuple[str, str, str]] = []
    queries: List[Tuple[str, str]] = []
    seen_pages = set()
    seen_queries = set()

    for rec in (as_record(s) for s in dispatcher.log):
        stype = _enum(rec.get("type"))
        agent = rec.get("agent_id") or "unknown"
        args = rec.get("args") or {}
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        if stype == "FILE_WRITE":
            page_id = args.get("filename")
            content = args.get("content")
            if page_id and content and page_id not in seen_pages:
                writes.append((agent, page_id, str(content)))
                seen_pages.add(page_id)
        elif stype == "LLM_CALL" and _enum(rec.get("status")) == SyscallStatus.SUCCESS.value:
            text = result.get("text")
            sid = (rec.get("syscall_id") or "")[:8] or str(len(writes))
            page_id = f"llm-{agent}-{sid}"
            if text and page_id not in seen_pages:
                writes.append((agent, page_id, str(text)))
                seen_pages.add(page_id)
        elif stype == "FILE_SEARCH":
            query = args.get("query") or result.get("query")
            if query and query not in seen_queries:
                queries.append((agent, str(query)))
                seen_queries.add(query)

    written = 0
    for agent, page_id, content in writes:
        await _set_quota_via_syscall(dispatcher, agent)
        syscall = await dispatcher.dispatch(
            agent, SyscallType.MEM_WRITE, page_id=page_id, content=content
        )
        if syscall.status == SyscallStatus.SUCCESS:
            written += 1

    read = 0
    for agent, query in queries:
        syscall = await dispatcher.dispatch(
            agent, SyscallType.MEM_READ, query_text=query
        )
        if syscall.status == SyscallStatus.SUCCESS:
            read += 1

    return {"mem_writes": written, "mem_reads": read, "candidates": len(writes)}


async def capture_session(
    pipeline_tasks: Sequence[str] = DEFAULT_PIPELINE_TASKS,
    assistant_questions: Sequence[str] = DEFAULT_ASSISTANT_QUESTIONS,
    driver: Optional[str] = None,
    persist_memory: bool = True,
    concurrent: bool = False,
) -> Dict[str, Any]:
    """Run real pipeline tasks + assistant questions on one live dispatcher."""
    from agents.kernel_assistant import ASSISTANT_PID, KernelAssistant
    from agents.pipeline import PipelineRunner
    from kernel.access_control import ResourceManager
    from kernel.memory import HashingEmbedder, set_embedder

    # Match API startup: open Chroma collections under hashing so capture cannot
    # hang on an unresponsive Ollama. The *benchmark* still prefers Ollama when
    # it replays the exported content; this only affects capture-time FILE_SEARCH
    # ranking, not the recorded query text or page contents.
    set_embedder(HashingEmbedder())

    driver = driver or _choose_driver()
    scratch = Path(tempfile.mkdtemp(prefix="agentos-real-capture-"))
    acl = AccessControl()
    dispatcher = SyscallDispatcher(
        access_control=acl,
        page_manager=PageManager(
            ram_budget_tokens=20000, chroma_path=str(scratch / "mem")
        ),
        filesystem=SemanticFS(
            access_control=acl,
            fs_root=str(scratch / "fs"),
            chroma_path=str(scratch / "fs_chroma"),
        ),
        quota_manager=QuotaManager(
            default_max_pages=200, default_max_calls_per_minute=40
        ),
        # Default ollama pool is 4 slots; concurrent pipelines + assistant
        # need enough Banker's capacity so overlapping LLM_CALLs are granted
        # rather than refused and serialised by fallback failure.
        resource_manager=ResourceManager(
            capacities={"groq": 30, "gemini": 15, "deepseek": 30, "ollama": 8}
        ),
        record_state=True,
    )

    assistant = KernelAssistant(dispatcher)
    await assistant.register(priority=1)
    await _set_quota_via_syscall(
        dispatcher, ASSISTANT_PID, max_pages=200, max_calls_per_minute=40
    )

    provenance: Dict[str, Any] = {
        "driver": driver,
        "capture_mode": "concurrent" if concurrent else "sequential",
        "capture_embedder": "hashing (indexing only; benches prefer Ollama on replay)",
        "pipeline_tasks": [],
        "assistant_questions": [],
        "scratch": str(scratch),
    }

    try:
        print(f"indexing documentation for assistant (driver={driver})...", flush=True)
        indexed = await assistant.index_documentation()
        provenance["docs_indexed"] = indexed.get("indexed")
        print(f"  indexed {indexed.get('indexed')} chunks", flush=True)

        runner = PipelineRunner(dispatcher)

        async def _one_pipeline(topic: str) -> Dict[str, Any]:
            print(f"pipeline: {topic[:72]}...", flush=True)
            started = time.time()
            try:
                result = await runner.run(topic, driver=driver, quotas=CAPTURE_QUOTAS)
                entry = {
                    "topic": topic,
                    "status": result.get("status"),
                    "run_id": result.get("run_id"),
                    "elapsed_s": round(time.time() - started, 3),
                    "stages": [
                        {"stage": s.get("stage"), "status": s.get("status")}
                        for s in (result.get("stages") or [])
                    ],
                }
            except Exception as exc:  # noqa: BLE001 — keep capturing after a failed task
                entry = {
                    "topic": topic,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.time() - started, 3),
                }
            print(f"  -> {entry['status']} ({entry['elapsed_s']}s) [{topic[:40]}]", flush=True)
            return entry

        async def _one_question(question: str) -> Dict[str, Any]:
            print(f"assistant: {question}", flush=True)
            started = time.time()
            reply = await assistant.answer(question, driver=driver)
            issued = [s.get("type") for s in reply.get("syscalls") or []]
            entry = {
                "question": question,
                "process_alive": reply.get("process_alive"),
                "elapsed_s": round(time.time() - started, 3),
                "syscalls": issued,
                "answer_preview": (reply.get("answer") or "")[:160],
            }
            print(f"  -> {len(issued)} syscalls ({entry['elapsed_s']}s)", flush=True)
            return entry

        if concurrent:
            print(
                "concurrent capture: 3 pipelines in parallel, assistant questions "
                "starting 1s later so they overlap in-flight LLM_CALLs",
                flush=True,
            )

            async def _questions_during() -> List[Dict[str, Any]]:
                await asyncio.sleep(1.0)
                entries: List[Dict[str, Any]] = []
                # one-at-a-time on pid "assistant" — two overlapping answer()
                # calls would collide on dispatcher._inflight_tasks[assistant]
                for question in assistant_questions:
                    entries.append(await _one_question(question))
                return entries

            pipe_entries, question_entries = await asyncio.gather(
                asyncio.gather(*[_one_pipeline(t) for t in pipeline_tasks]),
                _questions_during(),
            )
            provenance["pipeline_tasks"] = list(pipe_entries)
            provenance["assistant_questions"] = list(question_entries)
        else:
            for topic in pipeline_tasks:
                provenance["pipeline_tasks"].append(await _one_pipeline(topic))
            for question in assistant_questions:
                provenance["assistant_questions"].append(await _one_question(question))

        if persist_memory:
            print("persisting session artifacts into page memory...", flush=True)
            provenance["memory_persist"] = await _persist_session_memory(dispatcher)
            print(f"  -> {provenance['memory_persist']}", flush=True)

        return build_workload(
            dispatcher.log,
            snapshots=_snapshots_from_dispatcher(dispatcher),
            provenance=provenance,
        )
    finally:
        set_embedder(None)


def summarize(workload: Dict[str, Any]) -> str:
    procs = workload.get("scheduler", {}).get("processes") or []
    mem = workload.get("memory") or {}
    pages = mem.get("pages") or []
    accesses = mem.get("accesses") or []
    reads = sum(1 for a in accesses if a.get("op") == "read")
    writes = sum(1 for a in accesses if a.get("op") == "write")
    prov = workload.get("provenance") or {}
    lines = [
        f"source={workload.get('source')}  captured_at={workload.get('captured_at')}",
        f"syscalls={prov.get('syscall_count')}  types={prov.get('syscall_type_counts')}",
        f"scheduler processes={len(procs)}",
    ]
    if procs:
        bursts = [p["burst"] for p in procs]
        prios = {}
        for p in procs:
            prios[p["priority"]] = prios.get(p["priority"], 0) + 1
        lines.append(
            f"  burst range={min(bursts):.4f}-{max(bursts):.4f}s  "
            f"mean={sum(bursts)/len(bursts):.4f}s  "
            f"priority mix={dict(sorted(prios.items()))}"
        )
        contention = (workload.get("scheduler") or {}).get("contention") or {}
        if contention:
            lines.append(
                f"  contention: ready_queue_forms={contention.get('ready_queue_forms')}  "
                f"arrivals_while_in_flight={contention.get('arrivals_while_another_in_flight')}  "
                f"max_concurrent={contention.get('max_concurrent_intervals')}"
            )
    lines.append(f"memory pages={len(pages)}  accesses={len(accesses)} (writes={writes}, reads={reads})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="export a real captured syscall workload for the benches"
    )
    parser.add_argument(
        "--from-log",
        type=Path,
        help="JSON syscall dump (list, {syscalls,snapshots}, or pipeline result)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="run pipeline tasks + assistant questions against a live kernel",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="with --capture: run pipeline tasks in parallel (overlapping syscalls)",
    )
    parser.add_argument(
        "--driver",
        default=None,
        help="LLM driver for --capture (default: first available of ollama/gemini/groq)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="workload JSON path (default: real_captured.json, or "
        "real_captured_concurrent.json with --concurrent)",
    )
    parser.add_argument(
        "--no-memory-persist",
        action="store_true",
        help="skip the capture-time MEM_WRITE/MEM_READ pass over session artifacts",
    )
    args = parser.parse_args(argv)

    if args.capture and args.from_log:
        parser.error("choose one of --capture or --from-log")
    if not args.capture and not args.from_log:
        parser.error("one of --capture or --from-log is required")

    if args.concurrent and not args.capture:
        parser.error("--concurrent requires --capture")

    output = args.output
    if output is None:
        output = DEFAULT_CONCURRENT_WORKLOAD_PATH if args.concurrent else DEFAULT_WORKLOAD_PATH

    if args.from_log:
        syscalls, snapshots = load_syscall_dump(args.from_log)
        workload = build_workload(
            syscalls,
            snapshots=snapshots,
            provenance={"from_log": str(args.from_log.resolve())},
        )
    else:
        workload = asyncio.run(
            capture_session(
                driver=args.driver,
                persist_memory=not args.no_memory_persist,
                concurrent=args.concurrent,
            )
        )

    write_workload(workload, output)
    print(summarize(workload))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
