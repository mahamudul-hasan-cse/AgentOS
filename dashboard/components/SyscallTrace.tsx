"use client";

import { fetchSyscallLog, SyscallEntry } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";

function statusColor(status: string): string {
  switch (status) {
    case "success":
      return "text-emerald-400";
    case "error":
      return "text-rose-400";
    case "permission_denied":
      return "text-orange-400";
    case "not_implemented":
      return "text-slate-500";
    default:
      return "text-slate-400";
  }
}

function Row({ s }: { s: SyscallEntry }) {
  return (
    <div className="flex items-center gap-2 border-b border-slate-800/50 py-1 text-xs">
      <span className="w-28 shrink-0 font-semibold text-sky-300">{s.type}</span>
      <span className="w-24 shrink-0 truncate text-slate-400">{s.agent_id}</span>
      <span className={`w-36 shrink-0 ${statusColor(s.status)}`}>{s.status}</span>
      <span className="ml-auto shrink-0 tabular-nums text-slate-500">
        {s.latency_ms != null ? `${s.latency_ms.toFixed(1)}ms` : "—"}
      </span>
    </div>
  );
}

export function SyscallTrace() {
  const { data, error } = usePolling(fetchSyscallLog, 2000);
  const syscalls = data ?? [];

  return (
    <Panel title="Live Syscall Trace" subtitle="last 20 · polling 2s">
      {error && <p className="text-xs text-rose-400">backend unreachable: {error}</p>}
      {!error && syscalls.length === 0 && (
        <p className="text-xs text-slate-500">no syscalls yet</p>
      )}
      <div className="font-mono">
        {syscalls.map((s) => (
          <Row key={s.syscall_id} s={s} />
        ))}
      </div>
    </Panel>
  );
}
