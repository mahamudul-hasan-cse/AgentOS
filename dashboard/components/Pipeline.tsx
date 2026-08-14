"use client";

import { fetchPipelineStatus, PipelineStage, runPipeline } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useState } from "react";
import { Panel, StateBadge } from "./Panel";

const STAGES = ["researcher", "coder", "tester", "writer"];

function statusFor(stage: PipelineStage | undefined): string {
  return stage?.status ?? "pending";
}

export function Pipeline() {
  const [topic, setTopic] = useState(
    "Write a small Python function that adds two numbers and prints a demo result"
  );
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data } = usePolling(fetchPipelineStatus, running ? 1000 : 2500);
  const stages = data?.stages ?? [];

  async function start() {
    setError(null);
    setRunning(true);
    try {
      await runPipeline(topic);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <Panel
      title="Pipeline"
      subtitle="researcher -> coder -> tester -> writer"
    >
      <div className="space-y-3">
        <textarea
          className="h-20 w-full rounded border border-slate-800 bg-slate-950 p-2 text-xs text-slate-200 outline-none focus:border-sky-700"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          disabled={running}
        />
        <button
          className="rounded border border-sky-700 bg-sky-500/10 px-3 py-1.5 text-xs font-semibold text-sky-200 hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          onClick={start}
          disabled={running || !topic.trim()}
        >
          {running ? "running pipeline..." : "run pipeline"}
        </button>

        {error && <p className="text-xs text-rose-400">{error}</p>}

        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {STAGES.map((name) => {
            const stage = stages.find((s) => s.stage === name);
            return (
              <div
                key={name}
                className="rounded border border-slate-800 bg-slate-950/60 p-2"
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-200">
                    {name}
                  </span>
                  <StateBadge state={statusFor(stage)} />
                </div>
                <p className="truncate font-mono text-[10px] text-slate-500">
                  {stage?.agent_id ?? "-"}
                </p>
                {stage?.driver_used && (
                  <p className="mt-1 text-[10px] text-sky-300">
                    driver: {stage.driver_used}
                  </p>
                )}
                {stage?.file && (
                  <p className="mt-1 truncate text-[10px] text-emerald-300">
                    file: {stage.file}
                  </p>
                )}
                {stage?.produced && (
                  <p className="mt-1 line-clamp-2 text-[10px] text-slate-400">
                    {stage.produced}
                  </p>
                )}
                {(stage?.quota_events?.length ?? 0) > 0 && (
                  <p className="mt-1 text-[10px] text-amber-300">
                    quota events: {stage?.quota_events.length}
                  </p>
                )}
                {stage?.error && (
                  <p className="mt-1 line-clamp-2 text-[10px] text-rose-300">
                    {stage.error}
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {data?.tester && (
          <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs">
            <div className="font-semibold text-slate-200">
              Tester:{" "}
              <span className={data.tester.passed ? "text-emerald-300" : "text-rose-300"}>
                {data.tester.passed ? "PASS" : "FAIL"}
              </span>
            </div>
            <p className="mt-1 text-slate-500">
              exit={data.tester.exit_code ?? "-"} timeout={String(data.tester.timeout)} rejected={String(data.tester.rejected)}
            </p>
            {data.tester.stdout && (
              <pre className="mt-2 max-h-24 overflow-auto rounded bg-black/30 p-2 text-[10px] text-slate-300">
                {data.tester.stdout}
              </pre>
            )}
          </div>
        )}

        {data?.final_report && (
          <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
            <h3 className="mb-2 text-xs font-semibold text-slate-200">
              Final Report
            </h3>
            <pre className="whitespace-pre-wrap text-xs leading-5 text-slate-300">
              {data.final_report}
            </pre>
          </div>
        )}
      </div>
    </Panel>
  );
}
