# AIOS

[![tests](https://github.com/mahamudul-hasan-cse/AIOS/actions/workflows/tests.yml/badge.svg)](https://github.com/mahamudul-hasan-cse/AIOS/actions/workflows/tests.yml)

AIOS is a **kernel that governs real LLM agent execution through syscalls**, built for an Operating Systems course project around classic OS concepts — process scheduling, virtual memory/paging, the syscall dispatcher, IPC, access control, deadlocks, and resource allocation — treating each agent request as a process and the LLM context window as physical RAM.

The dashboard UI uses the working name **AgentOS-Lite**; the repository and environment variables use **AIOS**.

This project is inspired by [agiresearch/AIOS](https://github.com/agiresearch/AIOS) as an architectural reference, but is independently designed and implemented from scratch.

## Security & Identity Model

**Read this before interpreting any ACL, quota, or deadlock demo.**

Agent identity in AIOS is **caller-declared, not authenticated**. There is no login, token, or credential that proves who an agent is.

- Every syscall and most HTTP endpoints accept an **`agent_id`** field (or shell **`--agent`** flag) naming who the caller claims to be.
- The kernel's ACL then enforces **privilege rules based on that unverified claim**: a caller saying `agent_id=root` is treated as KERNEL-privileged; a caller saying `agent_id=mallory` is treated as USER-level.
- **Any client can claim to be `root` or any other agent.** Nothing stops a modified HTTP request or `python shell/repl.py --agent root` from acting as the built-in admin identity.
- Several dashboard **read** endpoints (`GET /scheduler/state`, `/memory/state/{agent}`, `/syscalls/log`, `/resources/state`, `/fs/list/{agent}`) expose kernel state **without going through the syscall ACL at all**, so they are world-readable to whoever can reach the API.

This is **appropriate for a local, single-user course kernel** where the goal is to *demonstrate* OS mechanisms (privilege levels, quotas, Banker's avoidance, wait-for graphs) in a controlled environment. It is **not a real security boundary** and must not be mistaken for authentication, authorization against hostile users, or multi-tenant isolation.

**Generated-code execution** (pipeline tester via `TOOL_CALL` / `python_sandbox`) uses a **course-project-grade subprocess safeguard** (AST deny-list, timeout, scratch directory, process-group kill) — not a hardened sandbox. See `kernel/sandbox.py` and the pipeline's `sandbox_review_note` output.

## The Syscall Dispatcher

Every agent–kernel interaction goes through a single choke point: `SyscallDispatcher.dispatch`. The dispatcher traps the call, applies its gates, routes to the owning subsystem, records status and `latency_ms`, and appends an entry to the syscall log. Scheduling, paging, ACL, quotas, Banker's resource claims, IPC, the semantic FS, and `TOOL_CALL` are **not independent side systems** — they are handlers and gates the dispatcher invokes on the way through.

Order of checks matters: **ENOSYS-before-EPERM**. An unknown syscall type returns `NOT_IMPLEMENTED` before privilege is considered, matching real trap semantics and remaining visible in the log. Known types then face ACL (KERNEL vs USER), per-agent page and LLM-call-rate quotas, and the resource/Banker's gate (provider slot claim before `LLM_CALL`) as applicable. Denials surface as `PERMISSION_DENIED`, `QUOTA_EXCEEDED`, or related statuses in the log — the log is the primary evidence channel.

Introspection syscalls (`PROC_LIST`, `MEM_STATE`, `RESOURCE_STATE`, `SYSCALL_LOG`) and the ring-buffer state recorder observe the same path: compact snapshots of real dispatcher completions, not a parallel observability stack.

Syscall-facing HTTP entry points (the kernel demo surface):

| Endpoint | Role |
|----------|------|
| `POST /generate` | Issues `LLM_CALL` through the dispatcher |
| `POST /scheduler/spawn` | `SPAWN_AGENT` → process table |
| `POST /scheduler/wait/{pid}` | `WAIT` / zombie reap |
| `POST /scheduler/terminate/{pid}` | `TERMINATE_AGENT` |
| `POST /scheduler/kill-tree/{pid}` | Subtree terminate via the same kill path |
| `GET /scheduler/state`, `GET /scheduler/tree` | Live process table / hierarchy |
| `POST /memory/write`, `POST /memory/query` | `MEM_WRITE` / `MEM_READ` → paging / faults |
| `GET /memory/state/{agent}` | RAM vs swap view of paging |
| `GET /syscalls/log` | Trace evidence |
| `GET /resources/state` | Banker's pool allocation state |
| `POST /resources/mode` | Toggle avoidance on/off |
| `GET /deadlock/graph`, `/deadlock/status`, `POST /deadlock/detect` | Wait-for graph + detect/recover |
| `GET/POST /quotas/{agent}` | Quota usage / KERNEL `SET_QUOTA` |
| `POST /fs/write`, `GET /fs/read`, `POST /fs/search`, `GET /fs/list/{agent}` | File syscalls + listing |
| `GET /replay/timeline`, `/replay/snapshot/{id}`, `/replay/diff/{a}/{b}` | Snapshot mechanism API |

LLM drivers (`GroqDriver`, `DeepSeekDriver`, `GeminiDriver`, `OllamaDriver`) with explicit timeouts and automatic **Groq/DeepSeek/Gemini → Ollama** fallback sit behind `LLM_CALL` on this path.

Phase-by-phase build history (same numbers as `git log --grep=Phase`): [`PROJECT_PLAN.md` § Build Roadmap](PROJECT_PLAN.md#9-build-roadmap-commit-history).

## Kernel subsystems the dispatcher routes to

Each of the following is a **gate or subsystem the dispatcher routes to**, not a standalone product feature. Evidence is the process table, memory state, wait-for graph, and syscall log — see [The Syscall Dispatcher](#the-syscall-dispatcher).

### Scheduling

FCFS, Round Robin, Priority, and MLFQ, plus **priority aging** and **MLFQ boost** starvation variants. The scheduler owns the ready queue and timelines for processes the dispatcher registers and updates.

### Memory paging

Paged context window (context window as RAM), FIFO / LRU / **Semantic-LRU**, ChromaDB swap, and page-fault reload through `MEM_READ` / `MEM_WRITE`. Resident-set and swap views are kernel state the dispatcher exposes via memory handlers and introspection.

### Deadlock and resource allocation

Banker's Algorithm avoidance (default) **or** detection + recovery when avoidance is off. Provider rate-limit **pools** are part of this resource gate — visible via shell `top` / `GET /resources/state` (no dashboard panel). Wait-for graph and mode toggle go through the deadlock / resources endpoints above.

### Process hierarchy

`SPAWN_AGENT`, `WAIT`, `TERMINATE_AGENT`, and kill-tree mutate a real process table: spawn / wait / zombies / orphans / kill-tree. Hierarchy is kernel truth read back through `/scheduler/tree` and related syscalls.

IPC (async message queue + shared blackboard) and the semantic FS (per-agent files with embedding search) are the same class of trapped surfaces — used heavily by the pipeline and assistant workloads below.

## Workloads that exercise the architecture

These are **real workloads that exercise the syscall architecture under load**, not headline features in their own right.

### Benchmarks (synthetic and real-data)

The algorithms are measured, not just implemented. Seeded suites are reproducible and citable; captured sessions are a separate validation path — see [`benchmarks/README.md`](benchmarks/README.md).

```bash
python -m benchmarks.scheduler_bench    # FCFS / RR / Priority (+aging) / MLFQ (+boost)
python -m benchmarks.memory_bench       # FIFO / LRU / Semantic-LRU / Random
python -m benchmarks.belady_bench       # capacity sweep, Belady's Anomaly
python -m benchmarks.cow_bench          # copy-on-write savings vs naive fork
```

**Synthetic (seeded):** statistical rigor — multi-seed scheduler profiles, memory traces, Belady capacity sweep, starvation/aging curves. Re-runs produce byte-identical JSON.

**Real captured execution:** live pipeline + assistant sessions exported from the syscall log (`--workload-source real`). No seed; one session; quote only as practice validation, not as a substitute for the seeded tables. Concurrent capture is required before a ready queue forms and algorithms can diverge.

**Starvation, and what it costs to fix.** Priority scheduling's textbook flaw, demonstrated and then repaired: problem → measurement → solution → measurement. Full write-up: [`benchmarks/README.md` §4](benchmarks/README.md#4-starvation-under-priority-scheduling-and-the-cost-of-fixing-it).

**Belady's Anomaly.** Capacity sweep validating FIFO/LRU/Semantic-LRU behavior, including the canonical FIFO anomaly at 3→4 frames. Full write-up: [`benchmarks/README.md` §3](benchmarks/README.md#3-beladys-anomaly-experiment).

### Copy-on-write

COW fork semantics in the page manager (Phase 17), with a dedicated bench for sharing vs naive fork — kernel-real fork behavior validated under load.

### Flagship pipeline

Researcher → Coder → Tester → Writer, kernel-governed via syscalls (`POST /pipeline/run`, shell `pipeline <task>`): spawn, ACL, quotas, `LLM_CALL`, `TOOL_CALL`, blackboard, and FS under staged multi-agent work.

### Kernel assistant

In-kernel chat agent with doc search (`/assistant/*`): a process that exercises `PROC_LIST` / `MEM_STATE` / `FILE_SEARCH` / `FILE_READ` / `LLM_CALL` (including ACL denial demos).

## Demo / accessibility layer

An **accessibility/demo layer over the kernel state above**. Screenshots are optional; do not lead a demo on the dashboard — prefer `strace`, `pipeline` / `run`, `mem`, `deadlock` / `mode`, and spawn/kill in the shell.

### Dashboard

Next.js dashboard in [`dashboard/`](dashboard/) — live panels (poll every ~2s unless noted):

- **Process table** — live scheduler state from `/scheduler/state`
- **Process tree** — live hierarchy from `/scheduler/tree`
- **Memory view** — RAM vs ChromaDB swap, COW accounting
- **Syscall trace** — last 20 syscalls
- **Deadlock** — wait-for graph, avoidance toggle, force detect/recover
- **Pipeline** — run and watch the flagship multi-agent workflow (shows which LLM driver served each stage)
- **Kernel assistant** — chat against indexed project docs

**Hidden from the dashboard UI** (code kept): Time Travel scrubber, Gantt chart (offline throwaway sim), and HealthBadge. Provider **rate-limit pools** remain visible via shell `top` / `GET /resources/state`.

See [`dashboard/README.md`](dashboard/README.md) for component details.

### Shell

Interactive REPL over the HTTP API — primary CLI demo ([`shell/README.md`](shell/README.md)):

```bash
uvicorn api.main:app --port 8000
python shell/repl.py                              # acts as KERNEL-privileged root
python shell/repl.py --agent alice                # act as a USER-level agent
python shell/repl.py --url http://localhost:8010  # custom API port
```

Commands include `ps`, `top`, `pstree`, `spawn`, `wait`, `kill`, `limits`, `mem`, `ls`, `cat`, `find`, `strace`, `deadlock`, `mode`, `run`, and **`pipeline <task>`** (research → code → test → report).

Remember: `--agent root` is a **declared identity**, not proof of privilege — see [Security & Identity Model](#security--identity-model).

### Time-travel scrubber

Ring-buffer kernel snapshots (`/replay/*`) remain implemented. The dashboard scrubber UI is **hidden** for a simpler demo; the snapshot API is still available.

## Quickstart

Two supported paths. **Neither needs an API key** — with no keys configured the kernel runs in Ollama-only mode, and with no Ollama either it falls back to a built-in offline embedder. Both paths are exercised in CI.

### Option A — Docker (one command)

Requires only Docker. Brings up Ollama, the FastAPI kernel, and the dashboard together, and pulls the `nomic-embed-text` embedding model on first run.

```bash
git clone https://github.com/mahamudul-hasan-cse/AIOS.git
cd AIOS
docker compose up
```

Then open **http://localhost:3000**. The API is on http://localhost:8000 (`GET /health` reports embedding backend and startup step status).

First run downloads the Ollama image and the embedding model (~1.5 GB total), so it takes a few minutes; subsequent runs start in seconds from cached volumes.

To use your own provider keys, keep them on the host — they are **mounted, never baked into the image**:

```bash
cp kernel/config.yaml.example kernel/config.yaml    # then edit it
AIOS_CONFIG=./kernel/config.yaml docker compose up
```

| | |
|---|---|
| dashboard | http://localhost:3000 |
| API | http://localhost:8000 |
| stop | `docker compose down` |
| stop and discard stored pages/models | `docker compose down -v` |

<details>
<summary><b>If port 8000 or 3000 is unavailable</b> (common on Windows)</summary>

Both published ports are overridable:

```bash
AIOS_API_PORT=8010 AIOS_DASHBOARD_PORT=3001 docker compose up
```

`AIOS_API_PORT` is the only knob you need for the API — the dashboard's API base URL is derived from it automatically.

On Windows this is worth knowing about even when nothing is listening on the port: Hyper-V reserves blocks of TCP ports, and 8000 frequently lands inside one. The symptom is a bind failure at startup —

```
Error response from daemon: ports are not available: ... bind:
An attempt was made to access a socket in a way forbidden by its access permissions.
```

Check the reserved blocks with:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

If your API port falls inside one of those ranges, pick another. (This was hit while verifying the stack on Windows 11 — 8000 sat inside a reserved 7942–8041 block.)
</details>

### Option B — Manual install

Requires Python 3.10+ and Node 18+.

```bash
git clone https://github.com/mahamudul-hasan-cse/AIOS.git
cd AIOS

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

uvicorn api.main:app --reload --port 8000
```

In a second terminal:

```bash
cd dashboard
npm install
set NEXT_PUBLIC_API_BASE=http://localhost:8000   # Windows cmd
# export NEXT_PUBLIC_API_BASE=http://localhost:8000   # macOS/Linux
npm run dev
```

Open http://localhost:3000. If the API runs on a non-default port, set `NEXT_PUBLIC_API_BASE` to match (the dashboard build default in code is `8012` when unset — always set this env var for manual dev unless your API listens on 8012).

**Optional extras**, none of which are required to run:

```bash
# real semantic embeddings + offline LLM fallback
ollama serve
ollama pull nomic-embed-text   # embeddings for the memory manager / semantic FS
ollama pull llama3             # offline/fallback LLM driver

# provider API keys (Groq / Gemini / DeepSeek)
cp kernel/config.yaml.example kernel/config.yaml   # then edit

# test runner
pip install -r requirements-dev.txt
python -m pytest tests/
```

Without `nomic-embed-text` the kernel automatically uses a built-in hashing embedder, so a fresh clone works offline with zero setup — but similarity then reflects shared vocabulary rather than meaning. Whichever backend is active is logged at startup and reported by `GET /health`.

Dependencies are split deliberately: `requirements.txt` is what the kernel needs to **run**, `requirements-dev.txt` is what you need to **verify** it (pytest). Nothing in the dev file is imported by `kernel/`, `api/`, `agents/`, or `shell/`.

### Configuration

`kernel/config.yaml` is gitignored because it holds real API keys; the tracked `kernel/config.yaml.example` is the template and is what Docker mounts by default.

One environment variable overrides Ollama host in config: **`AIOS_OLLAMA_HOST`** (Compose sets `http://ollama:11434`).

Startup embedding policy: **`AIOS_STARTUP_EMBEDDINGS=hashing`** (default) switches to the offline hashing embedder before optional doc indexing so boot never blocks on Ollama.

## Project Structure

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

## Limitations & Future Work

Honest constraints for this course kernel:

- **Not a real security boundary.** Agent identity is caller-declared; dashboard state reads bypass syscall ACL; this must not be mistaken for authentication, authorization against hostile users, or multi-tenant isolation. Generated-code execution is a course-project-grade subprocess safeguard — not a hardened sandbox. (See [Security & Identity Model](#security--identity-model).)
- **Semantic-LRU — negative result (synthetic).** At n=10, Semantic-LRU does not beat LRU on any seeded trace; embeddings are not inert on locality-bearing traces (beats random there), but the defensible claim is narrower than “beats LRU.” Remaining synthetic limits: 20 pages, 5 resident, 120 accesses, one corpus. Full write-up: [`benchmarks/README.md` §2](benchmarks/README.md#2-memory--page-replacement-benchmark-synthetic-seeded).
- **Real-data memory caveat (n=1).** Captured-session replay is consistent in direction with “Semantic-LRU fails to beat LRU,” but **n=1** — not a statistical confirmation. Do not fold real-data numbers into the seeded tables. ([`benchmarks/README.md` §6](benchmarks/README.md#6-real-captured-execution).)
- Checkpoint/restore, multi-tenant virtual kernels, and remote kernel mode were considered during planning but are out of scope for this course kernel.

## Tech stack

Python, FastAPI, Groq / DeepSeek / Gemini / Ollama, ChromaDB, Next.js, [agno](https://github.com/agno-agi/agno) (agent framework identity for example agents).
