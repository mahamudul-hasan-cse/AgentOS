# AgentOS-Lite — LLM Agent Operating System Simulator
### Project Blueprint & Build Roadmap

> Working name: **AgentOS-Lite** (rename freely — e.g. `neurokernel`, `pyllmos`, `agentkernel`)
> Inspired by: [AIOS (agiresearch/AIOS)](https://github.com/agiresearch/AIOS) — architecture concepts only, code built from scratch.
> Purpose: Operating Systems course project + portfolio centerpiece.

---

## 1. Project Goal

Build a simplified operating system simulator where **LLM agents are processes**. The kernel manages scheduling, memory (context window as RAM), syscalls, IPC, access control, and resource allocation — reimagining classic OS textbook concepts (Silberschatz) for the LLM-agent era, running entirely on free-tier LLM APIs (Groq, DeepSeek, Gemini) plus local Ollama as an offline fallback.

**Why this works well for grading + portfolio:**
- Every module maps 1:1 to a chapter in your OS course (process scheduling, memory management, file systems, deadlocks, IPC)
- It's a real, running system with a live dashboard — not just a slide deck
- The "Semantic-LRU" page replacement algorithm (below) is a genuinely original contribution you can write about in your report
- It doubles as a flagship GitHub project: "I built an OS kernel for LLM agents" is a much stronger portfolio line than another CRUD app

---

## 2. High-Level Architecture

```
                     ┌─────────────────────────────┐
                     │        Agents (agno)         │
                     │  UserAgent, ResearchAgent...  │
                     └──────────────┬───────────────┘
                                    │ syscalls only
                     ┌──────────────▼───────────────┐
                     │      SYSCALL DISPATCHER       │
                     │  LLM_CALL / MEM_R/W / TOOL /   │
                     │  SPAWN / IPC_SEND / IPC_RECV   │
                     └───┬───────┬───────┬───────┬────┘
              ┌──────────┘       │       │       └───────────┐
      ┌───────▼──────┐  ┌────────▼───┐ ┌─▼──────────┐ ┌──────▼───────┐
      │  SCHEDULER    │  │  MEMORY    │ │  ACCESS    │ │  IPC / MSG   │
      │ FCFS/RR/      │  │  MANAGER   │ │  CONTROL   │ │  QUEUE       │
      │ Priority/MLFQ │  │ (paging)   │ │  (ACL)     │ │ (blackboard) │
      └───────┬───────┘  └─────┬──────┘ └────────────┘ └──────────────┘
              │                │
      ┌───────▼───────┐  ┌─────▼──────────┐
      │  LLM DRIVER    │  │  ChromaDB       │
      │  (HAL layer)   │  │  "swap disk"    │
      │ Groq/DeepSeek/ │  │  for paged-out  │
      │ Gemini/Ollama  │  │  context chunks │
      └────────────────┘  └─────────────────┘
```

---

## 3. Module-by-Module Breakdown

### 3.1 Scheduler (`kernel/scheduler/`)
Each incoming agent request = a process with: `pid`, `arrival_time`, `estimated_burst` (proxy: expected token count), `priority`.

Implement and let the user switch between:
- **FCFS** — baseline
- **Round Robin** — with a token-based quantum (e.g., agent gets N tokens of generation before context-switching to next agent)
- **Priority Scheduling** — e.g., interactive/user-facing agents > background/batch agents
- **MLFQ (Multi-Level Feedback Queue)** — agents that use a lot of "burst" get demoted to a lower-priority queue

Expose a `/scheduler/gantt` endpoint so the dashboard can render a live Gantt chart.

### 3.2 Memory Manager (`kernel/memory/`)
Treat the **LLM context window as physical RAM**, measured in tokens.

- **Page** = a chunk of conversation history (e.g., one exchange or N tokens)
- **Page table** = maps an agent's full logical history to what's currently loaded in the physical context window
- **Page fault** = agent references something no longer in-window → triggers retrieval
- **Swap space** = ChromaDB — full chunk text + embedding stored here when paged out

Replacement algorithms to implement:
- **FIFO** — evict oldest page
- **LRU** — evict least-recently-used page
- **Semantic-LRU (your original contribution)** — instead of pure recency, evict the page with the **lowest embedding similarity to the current query**, using ChromaDB's vector search. This models "relevance" rather than just "time," and is worth a dedicated section in your report/paper.

On a page fault, do a ChromaDB similarity search to "page in" the most relevant evicted chunk — this is literally RAG, reframed as a page-fault handler.

### 3.3 Syscall Dispatcher (`kernel/syscalls/`)
Single choke point all agent-kernel interaction passes through — mirrors the trap/interrupt mechanism.

Syscall types: `LLM_CALL`, `MEM_READ`, `MEM_WRITE`, `TOOL_CALL`, `FILE_READ`, `FILE_WRITE`, `SPAWN_AGENT`, `IPC_SEND`, `IPC_RECV`

Every syscall gets logged with timestamp, agent id, and args → this log powers a live "strace"-style view in the dashboard, and is great evidence of system behavior for your report.

### 3.4 LLM Driver Layer / HAL (`kernel/drivers/`)
Hardware-abstraction-layer pattern — one `LLMDriver` interface, four implementations:

```python
class LLMDriver(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    def is_available(self) -> bool: ...

class GroqDriver(LLMDriver): ...      # fast, generous free tier — good default
class DeepSeekDriver(LLMDriver): ...  # cheap/free tier, strong reasoning
class GeminiDriver(LLMDriver): ...    # free tier via Google AI Studio
class OllamaDriver(LLMDriver): ...    # fully local, zero cost, offline fallback
```

Kernel picks a driver per-agent from config, and **automatically fails over to Ollama** if a cloud driver hits a rate limit or errors out — this "graceful degradation" behavior is a strong demo moment and matches your "offline-first AI" engineering angle for your portfolio.

*(Free-tier terms/limits change over time — verify current limits on each provider's console before finalizing your config.)*

### 3.5 IPC (`kernel/ipc/`)
- Simple async message queue for agent-to-agent messages
- A shared "blackboard" (in-memory dict, or Redis if you want persistence) for multi-agent collaboration — e.g., a ResearchAgent posts findings, a WriterAgent reads them

### 3.6 Access Control (`kernel/access_control/`)
Two privilege levels, enforced inside the syscall dispatcher:
- **Kernel-level agents**: can spawn other agents, read/write any agent's memory
- **User-level agents**: sandboxed — only their own memory/tools

### 3.7 Resource Allocation / Deadlock Avoidance
Model each provider's rate limit as a finite resource pool (e.g., "Groq: 30 req/min"). When multiple agents request the same provider concurrently, run a simplified **Banker's Algorithm** to avoid over-committing and causing all agents to stall ("deadlock") on a rate-limited provider — fall back to another driver instead.

### 3.8 Semantic File System (`kernel/filesystem/`)
A lightweight version of AIOS's LSFS idea: natural-language commands ("find my notes about scheduling") get translated into embedding search + file ops over a managed directory.

### 3.9 Dashboard (`dashboard/` — Next.js, matches your stack)
- Live process table (state: ready / running / waiting / terminated)
- Gantt chart of scheduler execution (recharts)
- Memory page table visualization — which chunks are "in RAM" vs. swapped to ChromaDB
- Live syscall trace feed
- Provider health panel (Groq/DeepSeek/Gemini/Ollama status + rate-limit remaining)

---

## 4. Suggested Directory Structure

```
agentos-lite/
├── kernel/
│   ├── scheduler/
│   │   ├── algorithms.py       # FCFS, RR, Priority, MLFQ
│   │   └── scheduler.py
│   ├── memory/
│   │   ├── page_manager.py
│   │   └── replacement.py      # FIFO, LRU, Semantic-LRU
│   ├── syscalls/
│   │   ├── dispatcher.py
│   │   └── types.py
│   ├── drivers/
│   │   ├── base.py
│   │   ├── groq_driver.py
│   │   ├── deepseek_driver.py
│   │   ├── gemini_driver.py
│   │   └── ollama_driver.py
│   ├── ipc/
│   │   └── message_queue.py
│   ├── access_control/
│   │   └── acl.py
│   ├── filesystem/
│   │   └── semantic_fs.py
│   └── config.yaml
├── agents/
│   └── example_agents.py       # built with agno
├── api/
│   └── main.py                  # FastAPI app exposing the kernel
├── dashboard/                    # Next.js frontend
├── tests/
├── docs/
│   ├── architecture.md
│   └── report.md                 # course submission write-up
└── README.md
```

---

## 5. Build Roadmap (adjust weeks to your actual deadline)

| Phase | Focus | Deliverable |
|---|---|---|
| 1 | LLM driver abstraction + FastAPI skeleton | `/generate` endpoint works with all 4 providers + fallback |
| 2 | Scheduler core + algorithms | CLI demo: submit N agent "processes," watch scheduling order |
| 3 | Memory manager + paging + ChromaDB swap | Context window fills, pages evict/reload correctly |
| 4 | Syscall dispatcher unifying everything | Every agent action goes through syscalls; log viewer works |
| 5 | Multi-agent concurrency + IPC | Two agents collaborate via blackboard/message queue |
| 6 | Access control + resource allocation | ACL enforced; rate-limit deadlock avoidance demoed |
| 7 | Dashboard UI | Live Gantt chart, memory view, syscall trace, provider health |
| 8 | Tests, docs, report, demo video, polish | Submission-ready + portfolio-ready |

---

## 6. Portfolio Polish Checklist
- [ ] Clean README with architecture diagram (reuse the ASCII diagram above or redraw it)
- [ ] Short demo GIF/video showing the dashboard live
- [ ] Blog post: "I built an OS kernel for LLM agents — here's what nobody tells you until it breaks" (fits your existing tagline)
- [ ] Technical report section explaining Semantic-LRU as your original contribution
- [ ] Clear citation of AIOS as architectural inspiration

---

## 7. Starter Prompts for Claude Code

Paste these into Claude Code once you've opened your project folder there. Do them roughly in order.

**Phase 1:**
> Read PROJECT_PLAN.md. Scaffold `kernel/drivers/` with a `base.py` defining an abstract `LLMDriver` class (`generate()`, `is_available()`), and four implementations: `GroqDriver`, `DeepSeekDriver`, `GeminiDriver`, `OllamaDriver`, each reading API keys from `kernel/config.yaml`. Add a FastAPI app in `api/main.py` with a single `/generate` endpoint that picks a driver by name and falls back to Ollama if the primary driver raises a rate-limit or connection error.

**Phase 2:**
> Now implement `kernel/scheduler/algorithms.py` with FCFS, Round Robin (token-based quantum), Priority, and MLFQ scheduling over a queue of agent "process" objects (pid, arrival_time, estimated_burst, priority). Add a `/scheduler/gantt` endpoint returning the execution timeline as JSON.

**Phase 3:**
> Implement `kernel/memory/page_manager.py` treating the LLM context window as physical RAM measured in tokens, with pages representing conversation chunks. Implement FIFO and LRU replacement in `replacement.py`, then add a `SemanticLRU` class that evicts the page with lowest cosine similarity to the current query embedding. Wire paged-out chunks into ChromaDB as swap storage, and implement a page-fault handler that retrieves the most relevant swapped chunk via similarity search.

*(Continue this pattern for phases 4–8, referencing the corresponding section of PROJECT_PLAN.md each time.)*

---

## 8. Notes
- Keep the real AIOS repo you downloaded as a **reference only** — don't copy code into your submission; cite it in your report as related work.
- Free-tier API limits for Groq/DeepSeek/Gemini change over time — check each provider's current dashboard before finalizing your fallback thresholds.
- If your course deadline is tighter than 8 weeks, cut scope by dropping IPC + deadlock avoidance (sections 3.5/3.7) first — scheduler + memory manager + syscalls + dashboard is enough for a strong grade on its own.
