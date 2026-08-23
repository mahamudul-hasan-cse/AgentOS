# AIOS (AgentOS-Lite) — Course Project Report

**Repository:** [mahamudul-hasan-cse/AIOS](https://github.com/mahamudul-hasan-cse/AIOS)  
**Working name (dashboard/shell):** AgentOS-Lite  
**Project status:** Feature-complete through Phase 31 (demo polish); feature freeze in effect  
**Related work (reference only):** [agiresearch/AIOS](https://github.com/agiresearch/AIOS) — architecture concepts; this codebase is original

---

## 1. Abstract

AIOS is a course-project **kernel that governs real LLM agent execution through syscalls**. Classic operating-system ideas — process scheduling, virtual memory / paging, the syscall trap, IPC, access control, resource allocation, and deadlock handling — are applied to LLM agents: each agent request is treated as a process, and the LLM context window is treated as physical RAM.

The system is not a slide-deck simulator of an OS. Live work (pipeline stages, assistant questions, memory reads/writes) goes through a single dispatcher, is logged with status and latency, and can be inspected via a shell and dashboard. Algorithms are evaluated with **seeded synthetic benches** for statistical rigor and **real captured syscall workloads** for practice validation.

---

## 2. Problem statement and goals

### 2.1 Problem

LLM multi-agent demos often hide OS concerns behind ad-hoc orchestration. Reviewers cannot see scheduling, paging, privilege checks, or resource contention as first-class mechanisms. A course OS project needs a **running system** where those mechanisms are visible and measurable.

### 2.2 Goals

1. Map textbook OS topics onto an LLM-agent runtime.
2. Make every meaningful agent–kernel interaction a **trapped syscall** (logged and gated).
3. Provide inspectable demos: shell (`strace`, `pipeline`, …) and dashboard.
4. Measure algorithms honestly — including negative results — with reproducible benches and optional real-data replay.

### 2.3 Non-goals

- Production authentication / multi-tenant isolation  
- Hardened sandbox for hostile code  
- Checkpoint/restore, remote kernel mode, or multi-tenant virtual kernels (considered in planning; out of scope)

---

## 3. System overview

### 3.1 Architecture (mental model)

```
Agents (pipeline, assistant, …)
        │  syscalls (+ HTTP)
        ▼
SyscallDispatcher.dispatch   ← single choke point
        │
        ├── Scheduler + process tree
        ├── Memory manager (paging, swap, COW)
        ├── Access control (ACL, quotas, Banker's)
        ├── IPC / blackboard
        └── Drivers (Groq / DeepSeek / Gemini / Ollama)
                    │
        FastAPI · Shell · Dashboard
```

### 3.2 The syscall dispatcher (core claim)

All agent–kernel work goes through `SyscallDispatcher.dispatch`:

- Trap → apply **gates** (ENOSYS-before-EPERM, ACL, quotas, Banker's resource claim) → route to subsystem → record **status** and **latency_ms** → append to the syscall log.
- Scheduling, paging, ACL, quotas, and resources are **not** separate side products; they are handlers/gates on this path.
- Primary evidence for demos and grading: **`GET /syscalls/log`** / shell **`strace`**.

### 3.3 Kernel subsystems (dispatcher routes)

| Topic | Implementation |
|--------|----------------|
| Scheduling | FCFS, Round Robin, Priority, MLFQ (+ aging / boost variants) |
| Memory | Paged context window; FIFO / LRU / Semantic-LRU; ChromaDB swap; COW |
| Access control | KERNEL vs USER ACL; page + LLM-call-rate quotas |
| Resources / deadlock | Banker's avoidance (default) or detect/recover when off |
| Processes | Spawn / wait / zombie / orphan / kill / kill-tree |
| IPC / FS | Message queue, blackboard, semantic file search |

### 3.4 Security note (required honesty)

Agent identity is **caller-declared** (`agent_id` / `--agent`), not authenticated. ACL demonstrates privilege rules *given* a claimed identity. Dashboard state reads intentionally bypass syscall ACL for visibility. This is appropriate for a local course kernel and **must not** be described as production security. Pipeline code execution uses course-grade sandbox safeguards (`kernel/sandbox.py`), not a hostile-code jail.

---

## 4. Flagship workload: research → code → test → report

The multi-agent **pipeline** registers stage agents as processes and drives work through syscalls (`SPAWN`, `LLM_CALL`, `TOOL_CALL`, blackboard, FS). Stages: researcher → coder → tester → writer.

**How this supports the course criteria:**

1. **Data/info via system calls** — live `strace` / Syscall Trace shows trapped calls.  
2. **No simulation for the live path** — real LLM providers (e.g. Groq) with measured latency; Ollama fallback when configured. (Gantt’s offline chart endpoint is an intentional exception and is hidden from the simplified dashboard.)  
3. **Optimization with real data** — scheduler/memory benches can replay **captured** pipeline/assistant sessions (`--workload-source real`), separate from synthetic seeded profiles.

The **kernel assistant** is a USER process that answers by issuing introspection/file/`LLM_CALL` syscalls (visible under each answer and in the global trace).

---

## 5. Evaluation and results

Evidence is kept in two buckets on purpose ([`benchmarks/README.md`](benchmarks/README.md)):

| Kind | Purpose | Flag / artifacts |
|------|---------|------------------|
| Synthetic, seeded | Statistical rigor (Belady, starvation, multi-seed memory) | default; `scheduler.json`, `memory.json`, … |
| Real captured | Practice check on live syscall timelines | `--workload-source real`; `*_real.json`, `*_real.png` |

### 5.1 Scheduling (synthetic)

Textbook-consistent behavior: MLFQ tends to win response time; on uniform bursts, FCFS can win wait/turnaround (preemption is overhead). Starvation under priority scheduling is demonstrated and mitigated with aging (problem → measure → fix → measure).

### 5.2 Memory / Semantic-LRU (synthetic) — honest negative

At n=10 seeds, **Semantic-LRU does not beat LRU** on the constructed traces. Embeddings are not inert on locality-bearing traces (beats random there), but the defensible claim is narrower than “beats LRU.” See benchmarks §2.

### 5.3 Belady’s anomaly (synthetic)

Capacity sweep validates stack vs non-stack behavior, including the canonical FIFO anomaly (documented in benchmarks §3).

### 5.4 Real captured execution

- **Sequential** capture often shows no ready-queue contention (algorithms can look identical on wait).  
- **Concurrent** capture produces overlapping syscalls; algorithms diverge; results align with the *uniform / long-burst* side of synthetic findings for that session. Still **n=1** — cite as validation, not as a replacement for seeded tables.  
- Real memory replay does **not** overturn “Semantic-LRU fails to beat LRU.”

Artifacts: `benchmarks/workloads/real_captured*.json`, `benchmarks/results/*_real.*`.

---

## 6. User surfaces

| Surface | Role |
|---------|------|
| **Shell** | Primary CLI demo: `strace`, `pipeline`, `ps`/`pstree`, `mem`, `deadlock`/`mode`, `run` |
| **Dashboard** | Live panels: process table/tree, memory, syscall trace, deadlock, pipeline, kernel assistant |
| **Dashboard (hidden for simpler demo)** | Time Travel scrubber, Gantt (offline sim), HealthBadge — code retained |
| **pytest / CI** | Regression coverage; Docker quickstart available |

Process table/tree **collapse terminated** processes by default so demos stay readable after many pipeline runs.

---

## 7. Technology stack

Python, FastAPI, ChromaDB, Groq / DeepSeek / Gemini / Ollama, Next.js dashboard, pytest, Docker Compose. Example agent identity via [agno](https://github.com/agno-agi/agno).

Default Groq model id was migrated off the retired `llama-3.1-8b-instant` to Groq’s recommended replacement (`openai/gpt-oss-20b`) after provider deprecation (Aug 2026).

---

## 8. Project timeline (abridged)

| Span | Content |
|------|---------|
| Phases 1–21 | Drivers → scheduler → memory → dispatcher → IPC → ACL/resources → dashboard → shell → embeddings → benches → process tree → deadlock → COW → Belady → starvation → CI/Docker → pipeline |
| Phase 27–29 | Docs: real-kernel framing; remove unbuilt panel promises; dispatcher-first README/PROJECT_PLAN |
| Phase 30 | Real-data export/benches, kernel assistant + dashboard chat, Groq model migration |
| Phase 31 | Dashboard demo polish (sectioned UI, hide low-priority panels, collapse terminated) |

Full commit subjects: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) build roadmap / `git log --grep=Phase`.

---

## 9. Limitations

1. Caller-declared identity — not a real security boundary.  
2. Semantic-LRU negative result (synthetic); real-data memory check is n=1.  
3. Course-grade code sandbox only.  
4. Out of scope: checkpoint/restore, multi-tenant virtual kernels, remote kernel mode.  
5. Minor ops debt (e.g. dashboard default API port vs Docker 8000; Gemini client package deprecation).

---

## 10. Conclusion

AIOS demonstrates that OS coursework can be grounded in a **live, syscall-governed LLM-agent kernel** rather than simulation-only slides. The dispatcher is the architectural center; the pipeline and assistant are workloads that exercise it; synthetic benches provide rigor; real captures check practice. The project is **feature-complete under freeze**, with documentation and demo UI aligned to that story.

---

## 11. How to reproduce a short viva demo

1. Start API + dashboard (e.g. API on port 8010 if 8000 is blocked on Windows).  
2. Run a short **Pipeline** task; watch **Syscall Trace**, **Process Tree**, **Memory**.  
3. Ask the **Kernel Assistant** a question; show syscall receipts.  
4. Open one `benchmarks/results/*_real.png` (or JSON) and contrast with synthetic benches.  
5. Optionally: shell `strace` / `pipeline` for the same evidence without the UI.

Quickstart details: [`README.md`](README.md). Benchmark methods: [`benchmarks/README.md`](benchmarks/README.md).

---

## 12. References

1. Project README and PROJECT_PLAN (this repository).  
2. Benchmark write-ups: `benchmarks/README.md` §§1–6.  
3. agiresearch/AIOS — related architectural reference only.  
4. Classic OS texts (scheduling, paging, Banker's, Belady) as mapped in the modules above.

---

*End of report.*
