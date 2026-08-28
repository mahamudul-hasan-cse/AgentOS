# Technical Report — AIOS (AgentOS-Lite)

**Course:** Operating Systems Project  
**Project title:** AIOS — A Syscall-Governed Kernel for LLM Agents  
**Repository / project page:** [https://github.com/mahamudul-hasan-cse/AIOS](https://github.com/mahamudul-hasan-cse/AIOS) (README = intro + demo video + skills)  
**Author:** Mahamudul Hasan  
**Student ID:** _[fill before portal upload]_  
**Status:** Feature-complete (Phase 31); feature freeze for demo honesty  

---

## 1. Short introduction

AIOS is a **working kernel**, not a slide-deck OS simulator. It treats each LLM agent request as a **process** and the LLM context window as **physical RAM**. Agent–kernel work is trapped through a single choke point — `SyscallDispatcher.dispatch` — which applies ACL, quotas, and Banker's resource claims, then routes to scheduling, paging, IPC, filesystem, and tool execution. A multi-agent **research → code → test → report** pipeline and a **kernel assistant** exercise that path with real LLM providers. Algorithms are measured with seeded synthetic benches and **real captured syscall workloads**.

This report focuses on the **challenges** of building that system, written in the required **STAR** format (Situation → Task → Action → Result).

---

## 2. Project goals (what “done” meant)

| Goal | How it shows up in the system |
|------|-------------------------------|
| Map classic OS topics onto LLM agents | Scheduler, paging, dispatcher, ACL, quotas, deadlock, process tree |
| Gather live work through system calls | Agents and mutating demos call `dispatcher.dispatch()`; evidence in `strace` / Syscall Trace |
| Avoid fake live paths | Real LLM drivers + real `subprocess` for tester code (AST reject only when unsafe) |
| Optimize / evaluate with real data | `benchmarks/real_data_export.py` + `--workload-source real` |
| Make demos inspectable | Shell + Next.js dashboard (process tree, memory, syscall log, pipeline, assistant) |

Non-goals: production auth, hostile-code jail, multi-tenant virtual kernels.

---

## 3. Challenges in STAR format

### Challenge A — Making the dispatcher the real center (not a thin wrapper)

**Situation**  
Early designs risked looking like “agents calling Python helpers,” with scheduling, memory, and ACL bolted on as separate demos. Course criteria required that **data and control** go through system calls, not ad-hoc bypasses.

**Task**  
Build one trap path that every meaningful agent action uses, with visible status and latency, so a grader can prove the kernel is in the loop.

**Action**  
- Implemented `SyscallDispatcher.dispatch` as the single choke point (trap → ENOSYS-before-EPERM → ACL → quotas → Banker's claim → handler → log).  
- Routed LLM calls, memory R/W, spawn/wait/terminate, FS, IPC, and `TOOL_CALL` through that path.  
- Exposed evidence via `GET /syscalls/log` and shell `strace`.  
- Documented honest exceptions: dashboard **read** endpoints and a few ops writes intentionally bypass ACL/dispatch for visibility (see README *Security & Identity Model*).

**Result**  
Live pipeline and assistant turns produce a dense, inspectable syscall trace. The architectural story is “dispatcher first,” not “feature checklist.” Remaining gaps (e.g. some GET state routes, Gantt queue replace, assistant registration) are documented rather than hidden — appropriate for a course kernel, not claimed as production isolation.

---

### Challenge B — “No simulation” vs offline schedule visualization

**Situation**  
The project needed live, real execution, but OS courses also expect Gantt-style schedule diagrams. A throwaway schedule run can look like the whole system is simulated.

**Task**  
Keep the **live agent path real** while still supporting schedule visualization without lying about what is simulated.

**Action**  
- Kept LLM generation on real drivers (Groq / DeepSeek / Gemini / Ollama) with measured `latency_ms`.  
- Implemented pipeline tester execution as real `subprocess.Popen` in `kernel/sandbox.py` (AST deny-list rejects unsafe code; no canned PASS path).  
- Implemented `POST /scheduler/gantt` as an offline timeline on throwaway `Scheduler` copies; documented it as offline visualization.  
- Hid Gantt and Time Travel from the simplified dashboard so the viva path emphasizes Trace / Tree / Memory / Pipeline / Assistant.

**Result**  
Demo narrative is clear: agents are governed live; Gantt is an optional offline chart tool. Reviewers can verify tester runs via sandbox exit codes and syscall log entries for `TOOL_CALL`.

---

### Challenge C — Semantic-LRU looked clever but did not beat LRU

**Situation**  
Semantic-LRU (evict by embedding distance) was a natural “AI twist” on paging. Early expectations assumed it would dominate classic LRU on paraphrased / clustered traces.

**Task**  
Evaluate replacement policies honestly for the course — including negative results — with reproducible methodology.

**Action**  
- Built seeded synthetic memory benches (`memory_bench.py`) across sequential, random, looping, clustered, and paraphrased traces.  
- Used real embeddings when Ollama/`nomic-embed-text` is available; documented hashing fallback as unfair for Semantic-LRU.  
- Reported multi-seed fault rates instead of cherry-picking a single run.

**Result**  
At the project’s n=10 synthetic setting, **Semantic-LRU does not beat LRU**. Embeddings are not useless (they beat random on locality-bearing traces), but the defensible claim is narrower than “semantic paging wins.” That honesty strengthens the report more than an inflated claim.

---

### Challenge D — Priority scheduling starvation and proving the fix

**Situation**  
Priority scheduling can starve low-priority work when high-priority arrivals keep arriving — textbook theory that is easy to assert and hard to show with numbers.

**Task**  
Demonstrate the failure mode, implement a mitigation, and re-measure so the “fix” is evidence-based.

**Action**  
- Added starvation growth / by-priority / tradeoff benches and charts under `benchmarks/`.  
- Implemented **priority aging** (and MLFQ boost variants) as scheduler policy options.  
- Compared wait/response behavior before vs after aging on the same seeded workloads.

**Result**  
Charts and JSON under `benchmarks/results/` show starvation under plain priority and improvement under aging — a complete loop: problem → measure → fix → measure.

---

### Challenge E — Synthetic benches alone were not enough for “real data”

**Situation**  
Seeded synthetic profiles are statistically useful but can be dismissed as “assumed” workloads. The course asked for optimization/evaluation that uses **captured** execution, not only synthetic assumptions.

**Task**  
Capture real syscall timelines from live sessions and replay them through the same bench harnesses.

**Action**  
- Built `benchmarks/real_data_export.py` to export from the live dispatcher log / capture sessions.  
- Committed workloads such as `benchmarks/workloads/real_captured.json` and `real_captured_concurrent.json` (`source: "real"`, syscall counts, provenance).  
- Wired `scheduler_bench.py` / `memory_bench.py` with `--workload-source real`.  
- Documented both paths in `benchmarks/README.md` §6 and the main README.

**Result**  
Concurrent real captures produce ready-queue contention and algorithm divergence (validation, n=1). Sequential captures can look identical across algorithms when there is no overlap — itself a useful finding. Real-data and synthetic paths are first-class, not hidden scripts.

---

### Challenge F — Demo UX vs kernel honesty under feature freeze

**Situation**  
After many phases, the dashboard accumulated panels (Time Travel, Gantt, health badges) that diluted the viva story and sometimes conflicted with “live kernel” messaging.

**Task**  
Polish the demo surface without adding new kernel features or overselling unfinished panels.

**Action**  
- Froze new kernel features.  
- Sectioned the UI (kernel state / observation / workloads).  
- Hid low-priority panels while keeping code.  
- Collapsed terminated processes by default in process table/tree.  
- Kept Kernel Assistant and multi-LLM driver labels visible.  
- Migrated Groq off the retired `llama-3.1-8b-instant` model to the provider’s current recommended id after a live API break.

**Result**  
A coherent 2–5 minute demo path: Pipeline → Syscall Trace → Process Tree → Memory → Assistant → one real-bench artifact. Documentation (README, PROJECT_PLAN, this report) matches what the UI actually shows.

---

## 4. System overview (for graders)

```
Agents (pipeline, assistant, …)
        │  syscalls (+ HTTP)
        ▼
SyscallDispatcher.dispatch   ← single choke point
        │
        ├── Scheduler + process tree
        ├── Memory manager (paging, swap, COW)
        ├── Access control (ACL, quotas, Banker's)
        ├── IPC / blackboard / semantic FS
        └── Drivers (Groq / DeepSeek / Gemini / Ollama)
                    │
        FastAPI · Shell · Dashboard
```

| Topic | Implementation |
|--------|----------------|
| Scheduling | FCFS, Round Robin, Priority, MLFQ (+ aging / boost) |
| Memory | Paged context; FIFO / LRU / Semantic-LRU; ChromaDB swap; COW |
| Access control | KERNEL vs USER ACL; page + LLM-call-rate quotas |
| Resources | Banker's avoidance (default) or detect/recover |
| Processes | Spawn / wait / zombie / orphan / kill-tree |
| Flagship workload | Researcher → coder → tester → writer pipeline |

**Identity honesty:** agent `agent_id` is caller-declared, not authenticated. ACL demos privilege *given a claim*. Dashboard reads intentionally bypass syscall ACL for visibility. Sandbox is course-grade, not a hostile jail.

---

## 5. Evaluation snapshot

| Evidence | Location |
|----------|----------|
| Synthetic scheduler / memory / Belady / starvation | `benchmarks/results/*.json`, `*.png` |
| Real captured workloads | `benchmarks/workloads/real_captured*.json` |
| Real-data bench outputs | `benchmarks/results/*_real*` |
| Methods | `benchmarks/README.md` |

Highlights: MLFQ often wins response time; FCFS can win wait on uniform bursts; Semantic-LRU fails to beat LRU on constructed traces; starvation under priority is measurable and mitigated by aging; concurrent real captures validate divergence under contention.

---

## 6. Skills demonstrated

- OS internals applied to a non-traditional workload (LLM agents)  
- Concurrent systems design (dispatcher gates, process table, resource pools)  
- Measurement discipline (seeded benches + real capture + honest negatives)  
- Full-stack delivery (Python kernel API, CLI shell, Next.js dashboard, CI/Docker)  
- Engineering communication (security model, limitations, feature freeze)

---

## 7. Limitations

1. Caller-declared identity — not a production security boundary.  
2. Semantic-LRU negative result; real memory check is typically n=1.  
3. Course-grade code sandbox only.  
4. Some observability/ops endpoints read or mutate state outside `dispatch()` (documented).  
5. Out of scope: checkpoint/restore, multi-tenant virtual kernels.

---

## 8. How to reproduce (viva / video)

1. Start API + dashboard (API often on **8010** if Windows blocks 8000).  
2. Run a short **Pipeline** task; watch **Syscall Trace**, **Process Tree**, **Memory**.  
3. Ask the **Kernel Assistant** one question; show syscall receipts.  
4. Open one `benchmarks/results/*_real.png` (or JSON) vs a synthetic result.  
5. Optional: shell `strace` / `pipeline`.

Details: [`README.md`](README.md). Demo script for the GitHub page video: [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md).

---

## 9. Submission mapping (portal)

| Portal requirement | Deliverable |
|--------------------|-------------|
| Technical report (STAR challenges) | This file (`COURSE_REPORT.md`) — export to PDF if the LMS requires a file upload |
| GitHub page (intro + 2–5 min video + skills) | **Repository README:** [mahamudul-hasan-cse/AIOS](https://github.com/mahamudul-hasan-cse/AIOS) |
| GitHub folder | Same repository (or Download ZIP) |

---

## 10. References

1. Repository README and PROJECT_PLAN.  
2. `benchmarks/README.md` §§1–6.  
3. [agiresearch/AIOS](https://github.com/agiresearch/AIOS) — architectural reference only; this codebase is original.  
4. Classic OS texts: scheduling, paging, Banker's algorithm, Belady’s anomaly.

---

*End of technical report.*
