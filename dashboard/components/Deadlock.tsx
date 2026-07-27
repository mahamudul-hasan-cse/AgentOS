"use client";

import { useCallback, useState } from "react";

import { API_BASE } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";

interface DeadlockStatus {
  deadlocked: boolean;
  cycle: string[];
  detection_runs: number;
  avoidance_enabled: boolean;
  interval_seconds: number;
  monitoring: boolean;
  recoveries: number;
}

interface GraphNode {
  agent_id: string;
  holds: Record<string, number>;
  waiting_on: string[];
}

interface DeadlockGraph {
  nodes: GraphNode[];
  edges: { from: string; to: string }[];
  waiting: Record<string, Record<string, number>>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

const fetchBoth = async (): Promise<[DeadlockStatus, DeadlockGraph]> =>
  Promise.all([
    getJSON<DeadlockStatus>("/deadlock/status"),
    getJSON<DeadlockGraph>("/deadlock/graph"),
  ]);

export function Deadlock() {
  const [nonce, setNonce] = useState(0);
  const [busy, setBusy] = useState(false);
  const { data, error } = usePolling(fetchBoth, 2000, nonce);
  const [status, graph] = data ?? [null, null];

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  const forceDetect = useCallback(async () => {
    setBusy(true);
    try {
      await fetch(`${API_BASE}/deadlock/detect?recover=true`, { method: "POST" });
    } catch {
      /* the polled state will surface any problem */
    } finally {
      setBusy(false);
      refresh();
    }
  }, [refresh]);

  const toggleAvoidance = useCallback(async () => {
    if (!status) return;
    setBusy(true);
    try {
      await fetch(`${API_BASE}/resources/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ avoidance_enabled: !status.avoidance_enabled }),
      });
    } catch {
      /* ignored: polled state reflects reality */
    } finally {
      setBusy(false);
      refresh();
    }
  }, [status, refresh]);

  const inCycle = new Set(status?.cycle ?? []);

  return (
    <Panel
      title="Deadlock"
      subtitle={
        status
          ? `${status.detection_runs} scans · ${status.recoveries} recoveries`
          : "—"
      }
    >
      {error && <p className="text-xs text-rose-400">backend unreachable: {error}</p>}

      {status && (
        <div className="space-y-3">
          {/* strategy + state */}
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded border px-2 py-0.5 text-xs ${
                status.avoidance_enabled
                  ? "border-sky-500/40 bg-sky-500/15 text-sky-300"
                  : "border-amber-500/40 bg-amber-500/15 text-amber-300"
              }`}
            >
              {status.avoidance_enabled
                ? "avoidance — Banker's Algorithm"
                : "detection + recovery"}
            </span>

            {status.monitoring && (
              <span className="flex items-center gap-1 text-[10px] text-slate-500">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
                scanning every {status.interval_seconds}s
              </span>
            )}
          </div>

          {status.monitoring && status.interval_seconds <= 10 && (
            <p className="text-[10px] text-slate-500">
              the scan auto-recovers a cycle within one interval — raise{" "}
              <code className="text-slate-400">deadlock.interval_seconds</code> in
              kernel/config.yaml to keep a deadlock observable for a demo
            </p>
          )}

          <div
            className={`rounded border px-3 py-2 ${
              status.deadlocked
                ? "border-rose-500/50 bg-rose-500/10"
                : "border-emerald-500/30 bg-emerald-500/5"
            }`}
          >
            {status.deadlocked ? (
              <>
                <div className="text-sm font-bold text-rose-300">** DEADLOCKED **</div>
                <div className="mt-1 font-mono text-xs text-rose-200">
                  {status.cycle.join(" → ")} → {status.cycle[0]}
                </div>
              </>
            ) : (
              <div className="text-sm text-emerald-300">
                clear — no cycle in the wait-for graph
              </div>
            )}
            {!status.deadlocked && status.avoidance_enabled && (
              <p className="mt-1 text-[10px] text-slate-500">
                avoidance is on, so cycles should never form — turn it off to demo
                detection
              </p>
            )}
          </div>

          {/* controls */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={forceDetect}
              disabled={busy}
              className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-sky-500 hover:bg-sky-500/10 disabled:opacity-50"
            >
              force detect + recover
            </button>
            <button
              onClick={toggleAvoidance}
              disabled={busy}
              className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-amber-500 hover:bg-amber-500/10 disabled:opacity-50"
            >
              avoidance: {status.avoidance_enabled ? "ON → turn off" : "OFF → turn on"}
            </button>
          </div>

          {/* wait-for graph */}
          {graph && graph.nodes.length === 0 && (
            <p className="text-xs text-slate-500">
              wait-for graph is empty — nothing held or awaited
            </p>
          )}

          {graph && graph.nodes.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs uppercase tracking-wider text-slate-500">
                wait-for graph
              </h3>
              <div className="space-y-1.5">
                {graph.nodes.map((node) => {
                  const cycled = inCycle.has(node.agent_id);
                  const holds = Object.entries(node.holds ?? {});
                  return (
                    <div
                      key={node.agent_id}
                      className={`rounded border px-2.5 py-1.5 font-mono text-xs ${
                        cycled
                          ? "border-rose-500/50 bg-rose-500/10"
                          : "border-slate-800 bg-black/20"
                      }`}
                    >
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span
                          className={
                            cycled ? "font-bold text-rose-300" : "font-semibold text-slate-100"
                          }
                        >
                          {node.agent_id}
                        </span>
                        {cycled && (
                          <span className="rounded border border-rose-500/50 px-1 text-[9px] text-rose-300">
                            in cycle
                          </span>
                        )}
                        <span className="text-slate-500">
                          holds{" "}
                          {holds.length
                            ? holds.map(([p, u]) => `${p}:${u}`).join(", ")
                            : "—"}
                        </span>
                      </div>
                      {node.waiting_on.length > 0 && (
                        <div className="mt-0.5 text-slate-400">
                          <span className="text-slate-600">└─ waits on </span>
                          {node.waiting_on.map((target, i) => (
                            <span key={target}>
                              {i > 0 && <span className="text-slate-600">, </span>}
                              <span
                                className={
                                  inCycle.has(target) ? "text-rose-300" : "text-sky-300"
                                }
                              >
                                → {target}
                              </span>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
