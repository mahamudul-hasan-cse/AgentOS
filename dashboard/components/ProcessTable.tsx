"use client";

import { fetchSchedulerState } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel, StateBadge } from "./Panel";

export function ProcessTable() {
  const { data, error } = usePolling(fetchSchedulerState, 2000);
  const processes = data?.processes ?? [];

  return (
    <Panel
      title="Process Table"
      subtitle={data?.algorithm ? `algorithm: ${data.algorithm}` : "—"}
    >
      {error && <p className="text-xs text-rose-400">backend unreachable: {error}</p>}
      {!error && processes.length === 0 && (
        <p className="text-xs text-slate-500">no processes in the queue</p>
      )}
      {processes.length > 0 && (
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th className="pb-2 pr-4">PID</th>
              <th className="pb-2 pr-4">State</th>
              <th className="pb-2 pr-4">Arrival</th>
              <th className="pb-2 pr-4">Remaining</th>
              <th className="pb-2">Prio</th>
            </tr>
          </thead>
          <tbody className="text-slate-300">
            {processes.map((p) => (
              <tr key={p.pid} className="border-t border-slate-800/60">
                <td className="py-1.5 pr-4 font-semibold text-slate-100">{p.pid}</td>
                <td className="py-1.5 pr-4">
                  <StateBadge state={p.state} />
                </td>
                <td className="py-1.5 pr-4 tabular-nums">{p.arrival_time}</td>
                <td className="py-1.5 pr-4 tabular-nums">
                  {p.remaining_burst}
                  <span className="text-slate-600"> / {p.estimated_burst}</span>
                </td>
                <td className="py-1.5 tabular-nums">{p.priority}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}
