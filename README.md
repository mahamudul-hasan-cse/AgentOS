# AIOS

[![tests](https://github.com/mahamudul-hasan-cse/AIOS/actions/workflows/tests.yml/badge.svg)](https://github.com/mahamudul-hasan-cse/AIOS/actions/workflows/tests.yml)

AIOS is an **LLM Agent Operating System simulator** built for an Operating Systems course project. It reimagines classic OS concepts — process scheduling, virtual memory/paging, syscalls, IPC, access control, deadlocks, and resource allocation — as the management layer for LLM agents, treating each agent request as a process and the LLM context window as physical RAM.

The dashboard UI uses the working name **AgentOS-Lite**; the repository and environment variables use **AIOS**.

This project is inspired by [agiresearch/AIOS](https://github.com/agiresearch/AIOS) as an architectural reference, but is independently designed and implemented from scratch.

## Security & Identity Model

**Read this before interpreting any ACL, quota, or deadlock demo.**

Agent identity in AIOS is **caller-declared, not authenticated**. There is no login, token, or credential that proves who an agent is.

- Every syscall and most HTTP endpoints accept an **`agent_id`** field (or shell **`--agent`** flag) naming who the caller claims to be.
- The kernel's ACL then enforces **privilege rules based on that unverified claim**: a caller saying `agent_id=root` is treated as KERNEL-privileged; a caller saying `agent_id=mallory` is treated as USER-level.
- **Any client can claim to be `root` or any other agent.** Nothing stops a modified HTTP request or `python shell/repl.py --agent root` from acting as the built-in admin identity.
- Several dashboard **read** endpoints (`GET /scheduler/state`, `/memory/state/{agent}`, `/syscalls/log`, `/resources/state`, `/fs/list/{agent}`) expose kernel state **without going through the syscall ACL at all**, so they are world-readable to whoever can reach the API.

This is **appropriate for a local, single-user kernel simulator** where the goal is to *demonstrate* OS mechanisms (privilege levels, quotas, Banker's avoidance, wait-for graphs) in a controlled environment. It is **not a real security boundary** and must not be mistaken for authentication, authorization against hostile users, or multi-tenant isolation.

**Generated-code execution** (pipeline tester via `TOOL_CALL` / `python_sandbox`) uses a **course-project-grade subprocess safeguard** (AST deny-list, timeout, scratch directory, process-group kill) — not a hardened sandbox. See `kernel/sandbox.py` and the pipeline's `sandbox_review_note` output.

## What is built (phases 1–23)

[`PROJECT_PLAN.md` §5](PROJECT_PLAN.md#5-build-roadmap-commit-history) lists every phase **using the same numbers as `git log --oneline --grep=Phase`**. Phases **1–21 are committed** (note: there is no Phase 8 tag). Phase **22** (Groq/Gemini timeouts, sandbox thread offload) is implemented but not yet committed. Phase **23** is this documentation sync. Highlights:

| Area | What you get |
|------|----------------|
| **LLM drivers** | `GroqDriver`, `DeepSeekDriver`, `GeminiDriver`, `OllamaDriver` with explicit timeouts and automatic **Groq/DeepSeek/Gemini → Ollama** fallback |
| **Scheduler** | FCFS, Round Robin, Priority, MLFQ, plus **priority aging** and **MLFQ boost** starvation variants; process tree with spawn/wait/zombies/kill-tree |
| **Memory** | Paged context window, FIFO/LRU/**Semantic-LRU**, ChromaDB swap, **copy-on-write** fork semantics |
| **Syscalls** | Full dispatcher choke point, ENOSYS-before-EPERM, syscall trace, quotas, `TOOL_CALL` sandbox |
| **IPC** | Async message queue + shared blackboard (used by pipeline and legacy collaborate demo) |
| **Access control** | KERNEL vs USER privilege, per-agent page and LLM-call-rate quotas |
| **Resources / deadlock** | Banker's Algorithm avoidance (default) **or** detection + recovery when avoidance is off |
| **Semantic FS** | Per-agent files with embedding search (`/fs/*`) |
| **Flagship pipeline** | Researcher → Coder → Tester → Writer, kernel-governed via syscalls (`POST /pipeline/run`) |
| **Kernel assistant** | In-kernel chat agent with doc search (`/assistant/*`, dashboard Chat panel) |
| **Time travel** | Ring-buffer kernel snapshots (`/replay/*`, dashboard scrubber) |
| **Shell** | Interactive REPL over the HTTP API — primary CLI demo ([`shell/README.md`](shell/README.md)) |
| **Dashboard** | Process table, Gantt, process tree, memory (with COW stats), syscall trace, deadlock panel, pipeline panel, assistant chat, embedding health badge |
| **Benchmarks** | Seeded scheduler, memory, Belady, and COW evaluation suites ([`benchmarks/README.md`](benchmarks/README.md)) |
| **Reliability** | Non-blocking startup with `/health`, configurable embedding fallback, driver timeouts, sandbox offloaded from the event loop |

Full phase-by-phase history: [`PROJECT_PLAN.md` §5](PROJECT_PLAN.md#5-build-roadmap-commit-history).

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

Without `nomic-embed-text` the kernel automatically uses a built-in hashing embedder, so a fresh clone works offline with zero setup — but similarity then reflects shared vocabulary rather than meaning. Whichever backend is active is logged at startup and reported by `GET /health` (dashboard **HealthBadge** shows Ollama vs hashing).

Dependencies are split deliberately: `requirements.txt` is what the kernel needs to **run**, `requirements-dev.txt` is what you need to **verify** it (pytest). Nothing in the dev file is imported by `kernel/`, `api/`, `agents/`, or `shell/`.

### Configuration

`kernel/config.yaml` is gitignored because it holds real API keys; the tracked `kernel/config.yaml.example` is the template and is what Docker mounts by default.

One environment variable overrides Ollama host in config: **`AIOS_OLLAMA_HOST`** (Compose sets `http://ollama:11434`).

Startup embedding policy: **`AIOS_STARTUP_EMBEDDINGS=hashing`** (default) switches to the offline hashing embedder before optional doc indexing so boot never blocks on Ollama.

## Shell (primary CLI demo)

```bash
uvicorn api.main:app --port 8000
python shell/repl.py                              # acts as KERNEL-privileged root
python shell/repl.py --agent alice                # act as a USER-level agent
python shell/repl.py --url http://localhost:8010  # custom API port
```

Commands include `ps`, `top`, `pstree`, `spawn`, `wait`, `kill`, `limits`, `mem`, `ls`, `cat`, `find`, `strace`, `deadlock`, `mode`, `run`, and **`pipeline <task>`** (research → code → test → report). See [`shell/README.md`](shell/README.md).

Remember: `--agent root` is a **declared identity**, not proof of privilege — see [Security & Identity Model](#security--identity-model).

## Dashboard

Next.js dashboard in [`dashboard/`](dashboard/) — live panels (poll every ~2s unless noted):

- **Time Travel** — scrub kernel snapshots from `/replay/timeline`
- **Process table** + **Gantt chart** (from seeded demo + `/scheduler/state`)
- **Process tree** — live hierarchy from `/scheduler/tree`
- **Memory view** — RAM vs ChromaDB swap, COW accounting
- **Syscall trace** — last 20 syscalls
- **Deadlock** — wait-for graph, avoidance toggle, force detect/recover
- **Pipeline** — run and watch the flagship multi-agent workflow
- **Kernel assistant** — chat against indexed project docs
- **Health badge** — active embedding backend (Ollama vs hashing) from `/health`

Provider **rate-limit pools** are visible in shell `top` (`GET /resources/state`), not yet as a dedicated dashboard panel.

See [`dashboard/README.md`](dashboard/README.md) for component details.

## Evaluation

The algorithms are measured, not just implemented. Everything is seeded, so results are reproducible and citable — see [`benchmarks/README.md`](benchmarks/README.md).

```bash
python -m benchmarks.scheduler_bench    # FCFS / RR / Priority (+aging) / MLFQ (+boost)
python -m benchmarks.memory_bench       # FIFO / LRU / Semantic-LRU / Random
python -m benchmarks.belady_bench       # capacity sweep, Belady's Anomaly
python -m benchmarks.cow_bench          # copy-on-write savings vs naive fork
```

### Starvation, and what it costs to fix

Priority scheduling's textbook flaw, demonstrated and then repaired: problem → measurement → solution → measurement. Full write-up: [`benchmarks/README.md` §4](benchmarks/README.md#4-starvation-under-priority-scheduling-and-the-cost-of-fixing-it).

### Belady's Anomaly

Capacity sweep validating FIFO/LRU/Semantic-LRU behavior, including the canonical FIFO anomaly at 3→4 frames. Full write-up: [`benchmarks/README.md` §3](benchmarks/README.md#3-beladys-anomaly-experiment).

## Tech stack

Python, FastAPI, Groq / DeepSeek / Gemini / Ollama, ChromaDB, Next.js, [agno](https://github.com/agno-agi/agno) (agent framework identity for example agents).
