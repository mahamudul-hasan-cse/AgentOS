# AIOS

AIOS is an **LLM Agent Operating System simulator** built for an Operating Systems course. It reimagines classic OS concepts — process scheduling, virtual memory/paging, syscalls, IPC, and access control — as the management layer for LLM agents, treating each agent request as a process and the LLM context window as physical RAM.

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
```

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

Make sure [Ollama](https://ollama.com) is running locally with a pulled model (used as the offline/fallback driver):

```bash
ollama serve
ollama pull llama3
```

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
