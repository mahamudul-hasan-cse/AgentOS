# AIOS demo video script (2–5 minutes)

Target length: **3:00–4:00**. Speak clearly; show the UI, not slides.

## Before recording

1. Start the API (e.g. port **8010** if 8000 is blocked) and the dashboard.
2. Confirm Groq (or fallback) works with a quick generate / assistant ping.
3. Clear or collapse old terminated processes so the tree stays readable.
4. Have one `benchmarks/results/*_real.png` (or JSON) ready in a second window.
5. Optional: open shell `strace` in a third pane for one cutaway.

## Shot list

| Time | What to show | What to say |
|------|----------------|-------------|
| 0:00–0:20 | GitHub README (top of repo) | “AIOS is a kernel that governs real LLM agents through system calls — not a slide simulator.” |
| 0:20–0:45 | Architecture one-liner on screen (dispatcher diagram in README) | “Every agent action goes through `SyscallDispatcher.dispatch` — ACL, quotas, then the subsystem.” |
| 0:45–2:15 | Dashboard: run **Pipeline** (short topic) | “Researcher, coder, tester, writer register as processes. Watch Syscall Trace fill with LLM_CALL, MEM_WRITE, TOOL_CALL.” |
| 2:15–2:45 | **Process Tree** + **Memory** | “Process tree is real kernel state. Memory panel shows paging against the context window as RAM.” |
| 2:45–3:30 | **Kernel Assistant** — one question | “The assistant is a USER process; answers come with syscall receipts you can see in the trace.” |
| 3:30–4:00 | Real-bench chart / `*_real.json` | “Optimization uses captured syscall workloads as well as synthetic benches — here’s a real-data result.” |
| 4:00–4:20 | Close on repo URL | “Code, STAR report, and benches are in the GitHub repo. Thanks.” |

## Topics that work well for Pipeline

- “Write a Python function that returns the nth Fibonacci number and tests it.”
- Keep it small so the tester finishes inside the video window.

## What not to dwell on

- Time Travel scrubber / Gantt (hidden; offline chart tooling).
- Long config editing or installing dependencies live.
- Claiming production-grade security — identity is caller-declared (one honest sentence is enough if asked).

## After recording

1. Upload to **YouTube** (unlisted is fine).
2. Open [`README.md`](../README.md) → section **Demo video** → replace the “video pending” block with your watch link or thumbnail embed:

```markdown
[![AIOS demo video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```

3. Commit and push `main`.
4. Portal **GitHub page** field: paste the **repository** URL  
   `https://github.com/mahamudul-hasan-cse/AIOS`  
   (README is the project page — intro + video + skills.)
