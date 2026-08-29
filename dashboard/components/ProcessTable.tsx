"use client";

import { fetchSchedulerState } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useMemo, useState } from "react";
import { Panel, StateBadge } from "./Panel";

/** How many recent terminated rows to preview when expanded. */
const TERMINATED_PREVIEW = 3;

export function ProcessTable() {
  const { data, error } = usePolling(fetchSchedulerState, 2000);
  const [showTerminated, setShowTerminated] = useState(false);

  const processes = data?.processes ?? [];

  const { active, terminated } = useMemo(() => {
    const active = processes.filter((p) => p.state !== "terminated");
    const terminated = processes.filter((p) => p.state === "terminated");
    return { active, terminated };
  }, [processes]);

  const terminatedShown = showTerminated
    ? terminated.slice(-TERMINATED_PREVIEW)
    : [];
  const terminatedHiddenExtra = showTerminated
    ? Math.max(0, terminated.length - TERMINATED_PREVIEW)
    : terminated.length;

  const rows = [...active, ...terminatedShown];

  return (
    <Panel
      title="Process Table"
      subtitle={data?.algorithm ? `algorithm: ${data.algorithm}` : "—"}
    >
      {error && (
        <p className="text-xs text-rose-400">backend unreachable: {error}</p>
      )}
      {!error && active.length === 0 && terminated.length === 0 && (
        <p className="text-xs text-slate-500">no processes in the queue</p>
      )}
      {!error && (active.length > 0 || showTerminated) && (
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
            {rows.map((p) => (
              <tr key={p.pid} className="border-t border-slate-800/60">
                <td className="py-1.5 pr-4 font-semibold text-slate-100">
                  {p.pid}
                </td>
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
      {terminated.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-800/80 pt-2 text-[11px] text-slate-500">
          <span>
            {terminated.length} terminated hidden
            {showTerminated && terminatedHiddenExtra > 0
              ? ` · showing last ${TERMINATED_PREVIEW}`
              : ""}
          </span>
          <button
            type="button"
            onClick={() => setShowTerminated((v) => !v)}
            className="rounded border border-slate-700 px-2 py-0.5 text-slate-400 hover:border-slate-500 hover:text-slate-200"
          >
            {showTerminated ? "hide terminated" : "show recent terminated"}
          </button>
        </div>
      )}
    </Panel>
  );
}
