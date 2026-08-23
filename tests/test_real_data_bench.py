"""Real-data workload export and the --workload-source real bench path.

These tests feed a constructed syscall log (no live provider) so CI stays
offline. The capture path that talks to Groq/Ollama is exercised by running
`python -m benchmarks.real_data_export --capture`, not by this file.
"""

from __future__ import annotations

import json

from benchmarks.memory_bench import run_benchmark as run_memory
from benchmarks.real_data_export import (
    build_workload,
    extract_memory_trace,
    extract_scheduler_processes,
    overlap_stats,
    process_index_from_snapshots,
    process_index_from_syscalls,
)
from benchmarks.scheduler_bench import ProcessSpec, run_benchmark as run_scheduler
from benchmarks.scheduler_bench import specs_from_real_workload


def _syscall(
    stype,
    agent,
    timestamp,
    latency_ms,
    args=None,
    result=None,
    status="success",
    syscall_id="x",
):
    return {
        "syscall_id": syscall_id,
        "agent_id": agent,
        "type": stype,
        "args": args or {},
        "timestamp": timestamp,
        "result": result,
        "status": status,
        "latency_ms": latency_ms,
    }


def sample_log():
    return [
        _syscall(
            "SPAWN_AGENT",
            "kernel",
            1000.0,
            1.0,
            args={"pid": "pipeline_aaa_researcher", "priority": 1, "estimated_burst": 2.0},
            result={"pid": "pipeline_aaa_researcher", "priority": 1, "estimated_burst": 2.0},
            syscall_id="spawn-r",
        ),
        _syscall(
            "SPAWN_AGENT",
            "kernel",
            1000.1,
            1.0,
            args={"pid": "pipeline_aaa_coordinator", "priority": 0, "estimated_burst": 4.0},
            result={"pid": "pipeline_aaa_coordinator", "priority": 0, "estimated_burst": 4.0},
            syscall_id="spawn-c",
        ),
        _syscall(
            "FILE_WRITE",
            "assistant",
            1000.2,
            12.0,
            args={"filename": "readme__paging.md", "content": "Paging treats the context window as RAM."},
            syscall_id="fw1",
        ),
        _syscall(
            "LLM_CALL",
            "pipeline_aaa_researcher",
            1001.0,
            1500.0,
            args={"prompt": "research fibonacci"},
            result={"driver_used": "groq", "text": "Use an iterative loop."},
            syscall_id="llm1",
        ),
        _syscall(
            "TOOL_CALL",
            "pipeline_aaa_tester",
            1003.0,
            80.0,
            args={"tool": "python_sandbox", "code": "print(55)"},
            result={"exit_code": 0},
            syscall_id="tool1",
        ),
        _syscall(
            "FILE_SEARCH",
            "assistant",
            1004.0,
            20.0,
            args={"query": "how does paging work here"},
            result={"query": "how does paging work here", "results": [{"filename": "readme__paging.md"}]},
            syscall_id="fs1",
        ),
        _syscall(
            "MEM_WRITE",
            "assistant",
            1005.0,
            8.0,
            args={"page_id": "readme__paging.md", "content": "Paging treats the context window as RAM."},
            result={"page": {"page_id": "readme__paging.md", "content": "Paging treats the context window as RAM."}},
            syscall_id="mw1",
        ),
        _syscall(
            "MEM_READ",
            "assistant",
            1006.0,
            9.0,
            args={"query_text": "how does paging work here"},
            result={"page": {"page_id": "readme__paging.md"}, "page_fault": False},
            syscall_id="mr1",
        ),
        _syscall(
            "LLM_CALL",
            "assistant",
            1007.0,
            2200.0,
            args={"prompt": "answer"},
            result={"driver_used": "groq", "text": "Paging is RAM for the context window."},
            syscall_id="llm2",
        ),
    ]


def test_extract_scheduler_uses_measured_latency_not_spawn_estimate():
    records = sample_log()
    index = process_index_from_syscalls(records)
    assert index["pipeline_aaa_researcher"]["priority"] == 1
    assert index["pipeline_aaa_coordinator"]["priority"] == 0

    procs = extract_scheduler_processes(records, index)
    assert [p["syscall_type"] for p in procs] == ["LLM_CALL", "TOOL_CALL", "LLM_CALL"]
    researcher = next(p for p in procs if p["agent_id"] == "pipeline_aaa_researcher")
    assert researcher["burst"] == 1.5  # 1500 ms, not the spawn estimate of 2.0
    assert researcher["priority"] == 1
    assert researcher["arrival_time"] == 0.0
    assistant = next(p for p in procs if p["agent_id"] == "assistant")
    assert assistant["burst"] == 2.2
    assert assistant["arrival_time"] == 6.0


def test_extract_memory_keeps_real_order_and_content():
    memory = extract_memory_trace(sample_log())
    page_ids = {p["page_id"] for p in memory["pages"]}
    assert "readme__paging.md" in page_ids
    ops = [(a["op"], a["syscall_type"]) for a in memory["accesses"]]
    assert ops == [
        ("write", "FILE_WRITE"),
        ("read", "FILE_SEARCH"),
        ("write", "MEM_WRITE"),
        ("read", "MEM_READ"),
    ]
    search = memory["accesses"][1]
    assert search["query"] == "how does paging work here"
    assert search["page_id"] == "readme__paging.md"


def test_snapshots_fill_priority_when_spawn_missing():
    snaps = [
        {
            "processes": [
                {"pid": "assistant", "priority": 1, "estimated_burst": 0.0},
            ]
        }
    ]
    index = process_index_from_snapshots(snaps)
    assert index["assistant"]["priority"] == 1


def test_scheduler_bench_loads_real_workload(tmp_path):
    workload = build_workload(sample_log())
    path = tmp_path / "real.json"
    path.write_text(json.dumps(workload), encoding="utf-8")

    results = run_scheduler(workload_source="real", workload_path=str(path))
    assert results["parameters"]["workload_source"] == "real"
    assert results["parameters"]["seed"] is None
    assert "starvation_sweep" not in results
    assert set(results["profiles"]) == {"real_captured"}
    algos = results["profiles"]["real_captured"]["algorithms"]
    assert set(algos) == {
        "fcfs",
        "round_robin",
        "priority",
        "priority_aging",
        "mlfq",
        "mlfq_boost",
    }
    assert algos["fcfs"]["makespan"] > 0


def test_specs_from_real_workload_round_trip():
    specs = specs_from_real_workload(build_workload(sample_log()))
    assert all(isinstance(s, ProcessSpec) for s in specs)
    assert len(specs) == 3


def test_memory_bench_loads_real_workload(tmp_path):
    workload = build_workload(sample_log())
    path = tmp_path / "real.json"
    path.write_text(json.dumps(workload), encoding="utf-8")

    results = run_memory(workload_source="real", workload_path=str(path))
    assert results["parameters"]["workload_source"] == "real"
    assert results["parameters"]["num_seeds"] == 1
    assert "paraphrase_lexical_baseline" not in results
    assert set(results["traces"]) == {"real_captured"}
    policies = results["traces"]["real_captured"]["policies"]
    assert set(policies) == {"fifo", "lru", "semantic_lru", "random"}
    for name in policies:
        assert 0.0 <= policies[name]["page_fault_rate"]["mean"] <= 1.0


def test_synthetic_scheduler_path_unchanged():
    results = run_scheduler(workload_source="synthetic")
    assert results["parameters"]["workload_source"] == "synthetic"
    assert results["parameters"]["seed"] == 20260726
    assert "starvation" in results["profiles"]
    assert "starvation_sweep" in results


def test_overlap_stats_sequential_has_no_ready_queue():
    procs = extract_scheduler_processes(sample_log(), process_index_from_syscalls(sample_log()))
    stats = overlap_stats(procs)
    assert stats["ready_queue_forms"] is False
    assert stats["arrivals_while_another_in_flight"] == 0
    assert stats["max_concurrent_intervals"] == 1
    workload = build_workload(sample_log())
    assert workload["scheduler"]["contention"]["ready_queue_forms"] is False


def test_overlap_stats_detects_overlapping_bursts():
    procs = [
        {"pid": "A", "arrival_time": 0.0, "burst": 5.0, "priority": 1},
        {"pid": "B", "arrival_time": 1.0, "burst": 2.0, "priority": 1},
        {"pid": "C", "arrival_time": 1.5, "burst": 2.0, "priority": 0},
    ]
    stats = overlap_stats(procs)
    assert stats["ready_queue_forms"] is True
    assert stats["arrivals_while_another_in_flight"] == 2
    assert stats["max_concurrent_intervals"] >= 3
