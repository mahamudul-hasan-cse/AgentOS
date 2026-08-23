# AIOS (AgentOS-Lite) — Project Blueprint

> **Status:** Phases **1–21** match tagged commits in `git log`. **Phase 22** (request-path reliability) is implemented in the working tree but not yet committed. **Phase 23** (this documentation sync) is the docs-only commit that follows. Phase numbers here are taken from commit messages, not renumbered.
>
> Repository name: **AIOS**. Dashboard/shell banner: **AgentOS-Lite**.
>
> Inspired by: [AIOS (agiresearch/AIOS)](https://github.com/agiresearch/AIOS) — architecture concepts only, code built from scratch.

---

## 1. Project Goal

Build a kernel that governs real LLM agent execution through syscalls, where **LLM agents are processes**. The kernel manages scheduling, memory (context window as RAM), the syscall dispatcher, IPC, access control, resource allocation, and deadlocks — reimagining classic OS textbook concepts for the LLM-agent era, running on free-tier LLM APIs (Groq, DeepSeek, Gemini) plus local Ollama as offline fallback.

**Why this works for grading + portfolio:**
- Every module maps to OS course topics (scheduling, paging, syscalls, deadlocks, IPC, access control)
- A real running system with dashboard and shell — not slides alone
- **Semantic-LRU** page replacement is an original algorithmic contribution with benchmark evidence
- Flagship **research → code → test → report** pipeline demonstrates kernel-governed multi-agent work

---

## 2. Security & Identity Model

**Agent identity is caller-declared, not authenticated.**

- Syscalls and HTTP mutation endpoints take an `agent_id` (shell: `--agent`) naming who the caller *claims* to be.
- ACL checks enforce KERNEL vs USER rules **against that unverified string**. There is no credential, session, or signature.
- Any HTTP client or shell user can pass `agent_id=root` and receive KERNEL privileges.
- Dashboard state reads (`GET /scheduler/state`, `/memory/state/{agent}`, `/syscalls/log`, etc.) bypass syscall ACL entirely for demo visibility.

This models **how** an OS would enforce privileges *given* a process identity — it does **not** model secure identity issuance. Appropriate for a local single-user course kernel; **not** production security.

Pipeline **generated-code execution** uses a restricted subprocess + AST deny-list (`kernel/sandbox.py`) — honest course-project safeguards, not a hostile-code sandbox.

---

## 3. High-Level Architecture

```
                     ┌─────────────────────────────┐
                     │   Agents (pipeline, chat,   │
                     │   example_agents via agno)  │
                     └──────────────┬──────────────┘
                                    │ syscalls (+ HTTP API)
                     ┌──────────────▼───────────────┐
                     │      SYSCALL DISPATCHER       │
                     │  LLM / MEM / TOOL / FILE /     │
                     │  SPAWN / IPC / DEADLOCK_* …    │
                     └───┬───────┬───────┬───────┬────┘
              ┌──────────┘       │       │       └───────────┐
      ┌───────▼──────┐  ┌────────▼───┐ ┌─▼──────────┐ ┌──────▼───────┐
      │  SCHEDULER    │  │  MEMORY    │ │  ACCESS    │ │  IPC / MSG   │
      │ + process tree│  │  MANAGER   │ │  CONTROL   │ │  QUEUE       │
      │ FCFS/RR/Prio/ │  │ + COW      │ │ ACL/quotas │ │ + blackboard │
      │ MLFQ (+fixes) │  │ + Semantic │ │ + Banker's │ │              │
      └───────┬───────┘  └─────┬──────┘ └────────────┘ └──────────────┘
              │                │
      ┌───────▼───────┐  ┌─────▼──────────┐  ┌────────────────────────┐
      │  LLM DRIVERS   │  │  ChromaDB       │  │  REPLAY / ASSISTANT /  │
      │  Groq/DeepSeek/│  │  swap + FS      │  │  PIPELINE (agents/)    │
      │  Gemini/Ollama │  │  embeddings     │  │                        │
      └────────────────┘  └─────────────────┘  └────────────────────────┘
              │
      ┌───────▼───────────────────────────────────────┐
      │  FastAPI (api/) · Shell (shell/) · Dashboard   │
      └───────────────────────────────────────────────┘
```

**Intentional exceptions to "everything through syscalls":**
- `POST /scheduler/gantt` runs a throwaway simulation for the Gantt chart (does not mutate live kernel state through the dispatcher).
- Several `GET` endpoints expose read-only kernel state directly for dashboard polling (no ACL).

---

## 4. Module Reference (as built)

| Module | Path | Notes |
|--------|------|-------|
| Scheduler | `kernel/scheduler/` | Six algorithms incl. aging/boost; `Scheduler` process tree |
| Memory | `kernel/memory/` | PageManager, FIFO/LRU/Semantic-LRU, COW, embeddings |
| Syscalls | `kernel/syscalls/` | Dispatcher, types, ENOSYS-before-EPERM, TOOL_CALL sandbox |
| Drivers | `kernel/drivers/` | Four LLM drivers; explicit HTTP timeouts; Ollama fallback |
| IPC | `kernel/ipc/` | Message queue + blackboard |
| Access control | `kernel/access_control/` | ACL, quotas, resource manager, deadlock detector |
| Semantic FS | `kernel/filesystem/` | Per-agent files + embedding search |
| Sandbox | `kernel/sandbox.py` | Pipeline tester code execution (demo-grade) |
| Replay | `kernel/replay/` | Ring-buffer snapshots, diff engine |
| Agents | `agents/` | Pipeline, kernel assistant, example agents |
| API | `api/main.py` | FastAPI endpoints, startup reliability, `/health` |
| Shell | `shell/repl.py` | Interactive REPL |
| Dashboard | `dashboard/` | Next.js live panels |
| Benchmarks | `benchmarks/` | Seeded evaluation suites |
| Tests | `tests/` | 135+ pytest cases |

---

## 5. Build Roadmap (commit history)

Phase titles below are copied from **`git log --grep=Phase`** commit subjects. There is no Phase 8 tag in the repository (the sequence jumps from 7 to 9).

| Phase | Git commit subject (abridged) | Status |
|-------|------------------------------|--------|
| 1 | LLM driver abstraction layer with fallback + README | Committed |
| 2 | Scheduler with FCFS, Round Robin, Priority, and MLFQ | Committed |
| 3 | Memory manager with paging + ChromaDB swap (FIFO/LRU/Semantic-LRU) | Committed |
| 3.8 | Semantic file system with per-agent scoping | Committed |
| 4 | Syscall dispatcher unifying all agent-kernel interaction | Committed |
| 5 | IPC + multi-agent collaboration via message queue and blackboard | Committed |
| 6 | Access control + resource allocation with deadlock avoidance | Committed |
| 7 | Live dashboard (Next.js) with CORS and /scheduler/state | Committed |
| 9 | Signal handling — terminate running agents, wired into live scheduler state | Committed |
| 10 | Per-agent resource quotas (memory pages + LLM call rate) | Committed |
| 11 | Interactive shell/REPL for kernel control | Committed |
| 12 | Time-travel replay with state snapshots and dashboard scrubber | Committed |
| 13 | Real semantic embeddings via Ollama with hashing fallback | Committed |
| 14 | Empirical benchmark suite for schedulers and page replacement | Committed |
| 15 | Process creation, hierarchy, and lifecycle (SPAWN_AGENT, zombies, orphans) | Committed |
| 16 | Deadlock detection, recovery, and wait-for graph visualization | Committed |
| 17 | Copy-on-write memory for fork(), completing SPAWN_AGENT | Committed |
| 18 | Belady's Anomaly experiment with capacity sweep | Committed |
| 19 | Starvation demonstration and aging-based mitigation | Committed |
| 20 | CI, Docker, and a reproducible quickstart | Committed |
| 21 | Kernel-governed multi-agent pipeline | Committed |
| 22 | Request-path reliability — Groq/Gemini explicit timeouts; sandbox TOOL_CALL offloaded via `asyncio.to_thread` | **Implemented; not yet committed** |
| 23 | Documentation sync — README, PROJECT_PLAN, shell README (security/identity model, pipeline command) | **This commit** |

---

## 6. Directory Structure (current)

```
AIOS/
├── kernel/
│   ├── scheduler/          # algorithms.py, scheduler.py
│   ├── memory/             # page_manager.py, replacement.py, embeddings.py
│   ├── syscalls/           # dispatcher.py, types.py
│   ├── drivers/            # groq, deepseek, gemini, ollama
│   ├── ipc/                # message_queue.py (blackboard)
│   ├── access_control/     # acl, quotas, resource_manager, deadlock_detector
│   ├── filesystem/         # semantic_fs.py
│   ├── replay/             # recorder.py
│   └── sandbox.py
├── agents/                 # pipeline.py, kernel_assistant.py, example_agents.py
├── api/                    # main.py
├── shell/                  # repl.py, README.md
├── dashboard/              # Next.js App Router UI
├── benchmarks/             # evaluation suites + results/
├── tests/                  # pytest
├── PROJECT_PLAN.md
└── README.md
```

---

## 7. Demo Surfaces

| Surface | Best for |
|---------|----------|
| **Dashboard** | Live visual demo: all panels, pipeline run, assistant chat, time travel |
| **Shell** | ACL/permission demo (`--agent mallory`), deadlock toggle, `pipeline <task>`, `strace` |
| **pytest** | Regression proof (135 tests) |
| **Benchmarks** | Report-grade algorithm measurements |

---

## 8. Portfolio Checklist

- [x] Architecture diagram (README + this document)
- [x] Benchmark write-ups with honest negative results (`benchmarks/README.md`)
- [x] Flagship multi-agent pipeline with inspectable syscall trace
- [x] Security/identity model documented prominently
- [ ] Short demo GIF/video
- [ ] Course report / blog post
- [ ] Migrate Gemini driver to `google.genai` (deprecated package warning)

---

## 9. Notes

- Keep the upstream [agiresearch/AIOS](https://github.com/agiresearch/AIOS) repo as **reference only** — cite as related work, do not copy code.
- Free-tier API limits change; verify provider consoles before demos.
- Provider rate-limit **pools** are shown in shell `top`, not yet as a dedicated dashboard panel (embedding health is shown via HealthBadge).
