# AgentOS Dashboard

A minimal Next.js (App Router + TypeScript + Tailwind) dashboard that polls the
FastAPI kernel and shows live panels:

1. **Process Table** — scheduler queue (`GET /scheduler/state`) with state badges
2. **Process Tree** — live hierarchy (`GET /scheduler/tree`)
3. **Memory View** — RAM vs ChromaDB swap for an `agent_id` (`GET /memory/state/{agent_id}`)
4. **Live Syscall Trace** — recent syscalls (`GET /syscalls/log?limit=20`)
5. **Deadlock** — wait-for graph, avoidance toggle, detect/recover
6. **Pipeline** — run the flagship multi-agent workflow (`POST /pipeline/run`)
7. **Kernel Assistant** — chat against indexed project docs (`POST /assistant/chat`)

All panels poll every ~2 seconds and degrade gracefully when the backend is down.

## Running it

The dashboard needs the FastAPI backend on **port 8000**. From the project root:

```bash
uvicorn api.main:app --reload --port 8000
```

Then, in a second terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000. Set `NEXT_PUBLIC_API_BASE` in `.env.local` if the API
uses a non-default port (see root `README.md`).
