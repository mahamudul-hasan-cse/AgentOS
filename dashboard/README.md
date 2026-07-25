# AgentOS-Lite Dashboard

A minimal Next.js (App Router + TypeScript + Tailwind) dashboard that polls the
FastAPI kernel and shows four live panels:

1. **Process Table** — the scheduler's current process queue (`GET /scheduler/state`) with color-coded state badges (ready / running / waiting / terminated).
2. **Gantt Chart** — a horizontal bar visualization (recharts) of the most recent `/scheduler/gantt` run, colored by pid.
3. **Memory View** — for a given `agent_id`, pages currently in RAM vs. pages swapped to ChromaDB (`GET /memory/state/{agent_id}`), with token counts.
4. **Live Syscall Trace** — a scrolling feed of recent syscalls (`GET /syscalls/log?limit=20`), color-coded by status, polled every 2 seconds.

All panels poll every 2 seconds and degrade gracefully when the backend is down.

## Running it

The dashboard needs the FastAPI backend running on **port 8000**. From the
project root, with the Python virtualenv active:

```bash
uvicorn api.main:app --reload --port 8000
```

Then, in a second terminal, from the `dashboard/` folder:

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000. The backend allows CORS from `http://localhost:3000`
(configured in `api/main.py`), and seeds a demo scheduler queue and a `demo`
agent's memory on startup so every panel shows data immediately.

## Configuration

The backend base URL defaults to `http://localhost:8000`. To point elsewhere,
set `NEXT_PUBLIC_API_BASE` before `npm run dev`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev
```
