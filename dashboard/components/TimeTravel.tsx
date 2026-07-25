"use client";

import { Panel } from "./Panel";
import { useTimeTravel } from "./TimeTravelContext";

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

export function TimeTravel() {
  const { timeline, selectedId, isLive, selectSnapshot, returnToLive } =
    useTimeTravel();

  const count = timeline.length;
  // the slider indexes into the timeline; the far right position means "live"
  const selectedIndex =
    selectedId === null
      ? count // one past the last snapshot == live
      : Math.max(0, timeline.findIndex((s) => s.snapshot_id === selectedId));
  const current = selectedId === null ? null : timeline[selectedIndex];

  const onChange = (value: number) => {
    if (value >= count) {
      returnToLive();
    } else {
      selectSnapshot(timeline[value].snapshot_id);
    }
  };

  return (
    <Panel
      title="Time Travel"
      subtitle={count > 0 ? `${count} snapshot${count === 1 ? "" : "s"}` : "—"}
    >
      {count === 0 && (
        <p className="text-xs text-slate-500">
          no snapshots yet — they are captured automatically as syscalls run
        </p>
      )}

      {count > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <span
              className={`flex items-center gap-1.5 text-xs ${
                isLive ? "text-emerald-300" : "text-amber-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  isLive ? "animate-pulse bg-emerald-400" : "bg-amber-400"
                }`}
              />
              {isLive ? "LIVE" : "HISTORY"}
            </span>
            {!isLive && (
              <button
                onClick={returnToLive}
                className="rounded border border-amber-500/50 px-2 py-0.5 text-xs text-amber-200 hover:bg-amber-500/20"
              >
                return to live
              </button>
            )}
          </div>

          <input
            type="range"
            min={0}
            max={count}
            step={1}
            value={selectedIndex}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full accent-sky-400"
            aria-label="scrub through kernel history"
          />

          <div className="flex justify-between text-[10px] text-slate-600">
            <span>{timeline[0] ? formatTime(timeline[0].timestamp) : ""}</span>
            <span>now</span>
          </div>

          <div className="rounded border border-slate-800 bg-black/20 px-3 py-2">
            {current ? (
              <>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-semibold text-slate-100">
                    {current.label}
                  </span>
                  <span className="shrink-0 text-xs text-slate-500">
                    #{current.snapshot_id} · {formatTime(current.timestamp)}
                  </span>
                </div>
                {current.syscall_id && (
                  <p className="mt-0.5 truncate text-[10px] text-slate-600">
                    triggered by syscall {current.syscall_id}
                  </p>
                )}
              </>
            ) : (
              <span className="text-sm text-slate-300">
                live — showing current kernel state
              </span>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}
