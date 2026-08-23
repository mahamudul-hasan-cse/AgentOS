"use client";

import {
  AssistantChatReply,
  AssistantSyscall,
  askAssistant,
  fetchAssistantStatus,
  restartAssistant,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useState } from "react";
import { Panel } from "./Panel";

interface Turn {
  role: "user" | "assistant";
  content: string;
  syscalls?: AssistantSyscall[];
}

function statusColor(status: string): string {
  switch (status) {
    case "success":
      return "text-emerald-400";
    case "permission_denied":
      return "text-orange-400";
    case "quota_exceeded":
      return "text-amber-400";
    case "error":
      return "text-rose-400";
    default:
      return "text-slate-400";
  }
}

/** The grounding view: exactly which syscalls produced the answer above. */
function SyscallReceipt({ syscalls }: { syscalls: AssistantSyscall[] }) {
  if (syscalls.length === 0) return null;
  return (
    <div className="mt-2 rounded border border-slate-800 bg-black/30 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
        grounded in {syscalls.length} syscall{syscalls.length === 1 ? "" : "s"}
      </div>
      <div className="font-mono">
        {syscalls.map((s) => (
          <div key={s.syscall_id} className="flex items-center gap-2 py-0.5 text-[11px]">
            <span className="w-24 shrink-0 font-semibold text-sky-300">{s.type}</span>
            <span className="min-w-0 flex-1 truncate text-slate-400" title={s.target ?? ""}>
              {s.target ?? "—"}
            </span>
            <span className={`w-32 shrink-0 ${statusColor(s.status)}`}>{s.status}</span>
            <span className="w-16 shrink-0 text-right tabular-nums text-slate-500">
              {s.latency_ms != null ? `${s.latency_ms.toFixed(1)}ms` : "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Chat() {
  const { data: status } = usePolling(fetchAssistantStatus, 2000);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The process being gone is a first-class state, not an error: killing
  // `assistant` from the shell is a supported demonstration.
  const dead = status != null && !status.alive;

  async function send() {
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setError(null);
    const history = turns.map((t) => ({ role: t.role, content: t.content }));
    setTurns((prev) => [...prev, { role: "user", content: message }]);
    setBusy(true);
    try {
      const reply: AssistantChatReply = await askAssistant(message, history);
      setTurns((prev) => [
        ...prev,
        { role: "assistant", content: reply.answer, syscalls: reply.syscalls },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function restart() {
    setError(null);
    try {
      await restartAssistant();
      setTurns((prev) => [
        ...prev,
        { role: "assistant", content: "Process 'assistant' restarted. Ask me anything." },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <Panel
      title="Kernel Assistant"
      subtitle={
        status
          ? `pid ${status.pid} · ${status.privilege} · ${status.indexed_documents} docs indexed`
          : "connecting…"
      }
    >
      <div className="flex h-full min-h-[360px] flex-col">
        <p className="mb-3 shrink-0 text-[11px] leading-relaxed text-slate-500">
          This assistant is itself process{" "}
          <span className="font-mono text-slate-300">assistant</span> in the
          table above — it answers by issuing syscalls through the kernel. Every
          read is listed under its answer and in the live syscall trace.
        </p>

        {dead && (
          <div className="mb-3 shrink-0 rounded border border-rose-500/40 bg-rose-500/10 p-3">
            <p className="text-xs font-semibold text-rose-300">
              process &lsquo;{status?.pid}&rsquo; is {status?.state ?? "gone"}
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
              The assistant was terminated, so it can no longer issue syscalls.
              Restart it here, or from the shell with{" "}
              <span className="font-mono text-slate-300">kill assistant</span>.
            </p>
            <button
              onClick={restart}
              className="mt-2 rounded border border-slate-700 bg-slate-800/60 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-700/60"
            >
              restart process
            </button>
          </div>
        )}

        <div className="mb-3 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
          {turns.length === 0 && !dead && (
            <p className="text-xs text-slate-500">
              Try: “what processes are running?” · “what did the starvation
              benchmark find?” · “what is in researcher&rsquo;s memory?”
            </p>
          )}
          {turns.map((t, i) => (
            <div key={i}>
              <div
                className={`text-xs leading-relaxed ${
                  t.role === "user" ? "text-slate-300" : "text-slate-200"
                }`}
              >
                <span
                  className={`mr-2 font-mono text-[10px] uppercase ${
                    t.role === "user" ? "text-slate-500" : "text-sky-400"
                  }`}
                >
                  {t.role === "user" ? "you" : "assistant"}
                </span>
                <span className="whitespace-pre-wrap">{t.content}</span>
              </div>
              {t.syscalls && <SyscallReceipt syscalls={t.syscalls} />}
            </div>
          ))}
          {busy && (
            <p className="text-xs text-slate-500">
              issuing syscalls<span className="animate-pulse">…</span>
            </p>
          )}
        </div>

        {error && (
          <p className="mb-2 shrink-0 text-xs text-rose-400">error: {error}</p>
        )}

        <div className="flex shrink-0 gap-2 border-t border-slate-800 pt-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") send();
            }}
            disabled={dead || busy}
            placeholder={
              dead ? "process terminated" : "ask about the running kernel…"
            }
            className="min-w-0 flex-1 rounded border border-slate-700 bg-black/40 px-2 py-1.5 font-mono text-xs text-slate-200 placeholder:text-slate-600 focus:border-sky-600 focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={send}
            disabled={dead || busy || input.trim() === ""}
            className="shrink-0 rounded border border-sky-700/60 bg-sky-500/10 px-3 py-1.5 text-xs font-semibold text-sky-200 hover:bg-sky-500/20 disabled:opacity-40"
          >
            send
          </button>
        </div>
      </div>
    </Panel>
  );
}
