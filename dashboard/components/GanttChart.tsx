"use client";

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { colorForPid, fetchSchedulerState, GanttSlice } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";

// One row per pid. A pid's (possibly non-contiguous) slices become an
// alternating stack of transparent "gap" segments and colored "duration"
// segments, so multiple bars land on the same row — e.g. under Round Robin a
// process that runs in several slices shows several bars on its single row.
interface Row {
  pid: string;
  slices: GanttSlice[];
  [segment: string]: unknown; // gap0, dur0, gap1, dur1, ...
}

function buildRows(timeline: GanttSlice[]): { rows: Row[]; maxSlices: number } {
  const byPid = new Map<string, GanttSlice[]>();
  for (const s of timeline) {
    const list = byPid.get(s.pid);
    if (list) list.push(s);
    else byPid.set(s.pid, [s]);
  }
  for (const list of byPid.values()) list.sort((a, b) => a.start - b.start);

  // order rows by each process's first appearance in time
  const pids = [...byPid.keys()].sort(
    (a, b) => byPid.get(a)![0].start - byPid.get(b)![0].start
  );
  const maxSlices = pids.reduce((m, p) => Math.max(m, byPid.get(p)!.length), 0);

  const rows: Row[] = pids.map((pid) => {
    const slices = byPid.get(pid)!;
    const row: Row = { pid, slices };
    let prevEnd = 0;
    slices.forEach((s, k) => {
      row[`gap${k}`] = s.start - prevEnd; // idle time on this row before the slice
      row[`dur${k}`] = s.end - s.start;
      prevEnd = s.end;
    });
    for (let k = slices.length; k < maxSlices; k++) {
      row[`gap${k}`] = 0;
      row[`dur${k}`] = 0;
    }
    return row;
  });

  return { rows, maxSlices };
}

function GanttTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const row: Row = payload[0].payload;
  return (
    <div className="rounded border border-slate-700 bg-[#0d1220] px-2 py-1 text-xs text-slate-200">
      <div className="font-semibold" style={{ color: colorForPid(row.pid) }}>
        {row.pid}
      </div>
      {row.slices.map((s, i) => (
        <div key={i} className="text-slate-400">
          {s.start} → {s.end} ({s.end - s.start}t)
        </div>
      ))}
    </div>
  );
}

export function GanttChart() {
  const { data, error } = usePolling(fetchSchedulerState, 2000);
  const { rows, maxSlices } = buildRows(data?.timeline ?? []);

  const bars = [];
  for (let k = 0; k < maxSlices; k++) {
    bars.push(
      <Bar
        key={`gap${k}`}
        dataKey={`gap${k}`}
        stackId="t"
        fill="transparent"
        isAnimationActive={false}
      />
    );
    bars.push(
      <Bar
        key={`dur${k}`}
        dataKey={`dur${k}`}
        stackId="t"
        isAnimationActive={false}
        radius={2}
      >
        {rows.map((row) => (
          <Cell key={row.pid} fill={colorForPid(row.pid)} />
        ))}
      </Bar>
    );
  }

  return (
    <Panel title="Gantt Chart" subtitle="last schedule run · one row per process">
      {error && <p className="text-xs text-rose-400">backend unreachable: {error}</p>}
      {!error && rows.length === 0 && (
        <p className="text-xs text-slate-500">no timeline yet — run /scheduler/gantt</p>
      )}
      {rows.length > 0 && (
        <ResponsiveContainer width="100%" height={Math.max(160, rows.length * 40)}>
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 4, right: 16, bottom: 4, left: 8 }}
          >
            <XAxis
              type="number"
              stroke="#475569"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              allowDecimals={false}
            />
            <YAxis
              type="category"
              dataKey="pid"
              stroke="#475569"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              width={40}
            />
            <Tooltip content={<GanttTooltip />} cursor={{ fill: "#1e293b55" }} />
            {bars}
          </BarChart>
        </ResponsiveContainer>
      )}
    </Panel>
  );
}
