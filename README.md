# AIOS

AIOS is an **LLM Agent Operating System simulator** built for an Operating Systems. It reimagines classic OS concepts — process scheduling, virtual memory/paging, syscalls, IPC, and access control — as the management layer for LLM agents, treating each agent request as a process and the LLM context window as physical RAM.

This project is inspired by [agiresearch/AIOS](https://github.com/agiresearch/AIOS) as an architectural reference, but is independently designed and implemented from scratch.

## Features implemented so far

**Phase 1 — LLM driver abstraction layer**
- A single `LLMDriver` interface (`kernel/drivers/base.py`) with four hardware-abstraction-layer-style implementations: `GroqDriver`, `DeepSeekDriver`, `GeminiDriver`, `OllamaDriver`
- Config-driven API keys/models loaded from `kernel/config.yaml`
- A FastAPI `/generate` endpoint (`api/main.py`) that dispatches to the requested driver and **automatically falls back to Ollama** if the primary driver hits a rate limit or connection error

## Setup

```bash
git clone https://github.com/mahamudul-hasan-cse/AIOS.git
cd AIOS

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# optional: test runner + headless-browser dashboard verification
pip install -r requirements-dev.txt
python -m playwright install chromium chromium-headless-shell
```

Dependencies are split deliberately: `requirements.txt` is what the kernel needs
to **run**, `requirements-dev.txt` is what you need to **verify** it (pytest,
playwright). Nothing in the dev file is imported by `kernel/`, `api/`, `agents/`
or `shell/`.

`kernel/config.yaml` is gitignored (it holds real API keys). Copy the template and fill in your keys:

```bash
cp kernel/config.yaml.example kernel/config.yaml   # then edit kernel/config.yaml
```

The file has this structure:

```yaml
groq:
  api_key: "..."
  model: "llama-3.1-8b-instant"

gemini:
  api_key: "..."
  model: "gemini-1.5-flash"

deepseek:
  api_key: "..."
  model: "deepseek-chat"

ollama:
  host: "http://localhost:11434"
  model: "llama3"
```

Make sure [Ollama](https://ollama.com) is running locally with a pulled model (used as the offline/fallback driver), plus an embedding model used by the memory manager and semantic file system:

```bash
ollama serve
ollama pull llama3             # offline/fallback LLM driver
ollama pull nomic-embed-text   # real semantic embeddings
```

The embedding model is optional: without it the kernel automatically falls back to a built-in hashing embedder, so a fresh clone works offline with zero setup — but similarity then reflects shared vocabulary rather than meaning. Whichever backend is active is logged at startup.

Start the API server:

```bash
uvicorn api.main:app --reload
```

## Dashboard

A Next.js (App Router + TypeScript + Tailwind) dashboard in [`dashboard/`](dashboard/)
gives a live, terminal-styled view of the kernel: a process table with
color-coded state badges, a recharts Gantt chart of the last schedule run, a
memory panel showing pages in RAM vs. swapped to ChromaDB, and a live syscall
trace. All panels poll the backend every 2 seconds.

Run the FastAPI backend on port 8000, then in a second terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000. The backend enables CORS for `http://localhost:3000`
and seeds demo scheduler/memory state on startup so the panels are populated
immediately. See [`dashboard/README.md`](dashboard/README.md) for details.

## Evaluation

The algorithms are measured, not just implemented. Everything is seeded, so
results are reproducible and citable — see
[`benchmarks/README.md`](benchmarks/README.md) for the full write-up, raw JSON
and charts.

```bash
python -m benchmarks.scheduler_bench    # FCFS / RR / Priority (+aging) / MLFQ (+boost)
python -m benchmarks.memory_bench       # FIFO / LRU / Semantic-LRU / Random
python -m benchmarks.belady_bench       # capacity sweep, Belady's Anomaly
python -m benchmarks.cow_bench          # copy-on-write savings vs naive fork
```

### Starvation, and what it costs to fix

Priority scheduling's textbook flaw, demonstrated and then repaired:
problem → measurement → solution → measurement.

- **The flaw is real and averages hide it.** On a workload with a saturating
  stream of priority-0 arrivals, plain Priority posts the **best average
  waiting time of any algorithm (20.3)** while leaving the lowest-priority
  processes with the **worst starvation gap (84)**. Reporting the mean alone
  would have called it the winner.
- **It is starvation, not just a long wait.** Lengthening the stream 8×
  multiplies the worst low-priority wait by **4.7** under Priority (52 → 243)
  but only **1.3×** under Priority+Aging (52 → 69). Unbounded vs. bounded.
- **The fixes are added as variants**, `priority_aging` and `mlfq_boost`, so
  the originals stay measurable next to them.
- **The fix is not free, and the cost is explainable.** Bounding the wait costs
  the high-priority stream **+42.3 average waiting time**, which is almost
  exactly the victims' total burst — you cannot bound the low-priority wait
  without moving that work earlier, and moving it earlier is what it costs.
- **Aging is a dial, not a constant.** As its interval → 0 it degenerates to
  FCFS; as it → ∞ it degenerates to plain Priority, and the sweep hits both
  endpoints exactly. At an interval of 2.0 it is numerically identical to FCFS
  on all four profiles — a "fix" that deletes the thing it fixes. The sweep,
  not the chosen default, is the result.

Full write-up, tables and charts:
[`benchmarks/README.md` §4](benchmarks/README.md#4-starvation-under-priority-scheduling-and-the-cost-of-fixing-it).

### Belady's Anomaly

Adding memory should never make a page-replacement policy *worse* — except that
for some policies it can. This experiment sweeps RAM capacity as an independent
variable and looks for steps where an extra frame *raises* the fault rate.

- **The canonical reference string reproduces the textbook exactly**, run
  through the real `PageManager`: **FIFO 9 → 10 faults** at 3 → 4 frames (the
  anomaly), **LRU 10 → 8** (immune). Matching the published counts on the
  published input validates the kernel's own replacement implementations.
- **The broad sweep found zero anomalies** across 4 policies × 5 traces ×
  capacities 2–10 × 10 seeds. **LRU** showing none is the expected self-check —
  it is a stack algorithm, so an anomaly there would have meant a bug in our
  code. **FIFO** is therefore confirmed *in principle but not triggered by our
  workloads*.
- **Semantic-LRU showed no anomaly — a negative result, not proof of immunity.**
  It has no stack property, so nothing forbids one; this is absence of evidence
  over one workload set.
- Note the sweep runs on `PolicySim`, an in-memory replica of the policies
  (ChromaDB write throughput makes the live path ~8s per cell), cross-validated
  against the real kernel on the canonical string. The
  [full write-up](benchmarks/README.md#3-beladys-anomaly-experiment) covers what
  that does and does not license you to conclude.

A methodology note worth repeating: a 5-seed run reported an anomaly that
**vanished at 10 seeds**. The threshold for calling a step "systematic" was
tightened from >50% to ≥75% of seeds as a result, and every reported anomaly
now carries its seed count.

## Roadmap

Remaining phases from `PROJECT_PLAN.md`:

- **Scheduler** — FCFS, Round Robin (token-based quantum), Priority, and MLFQ scheduling over agent "processes"; `/scheduler/gantt` endpoint
- **Memory Manager** — context window as paged RAM, FIFO/LRU/Semantic-LRU replacement, ChromaDB as swap storage, page-fault handling via similarity search
- **Syscall Dispatcher** — single choke point for all agent-kernel interaction (`LLM_CALL`, `MEM_READ/WRITE`, `TOOL_CALL`, `SPAWN_AGENT`, `IPC_SEND/RECV`), with full syscall logging
- **IPC** — async message queue and shared blackboard for multi-agent collaboration
- **Access Control** — kernel-level vs. user-level agent privileges enforced in the syscall dispatcher
- **Resource Allocation / Deadlock Avoidance** — per-provider rate-limit pools with a simplified Banker's Algorithm
- **Semantic File System** — natural-language commands translated into embedding search + file ops
- **Dashboard** — live process table, Gantt chart, memory page view, syscall trace, provider health panel

## Tech stack

Python, FastAPI, Groq / DeepSeek / Gemini / Ollama, ChromaDB, [agno](https://github.com/agno-agi/agno)
