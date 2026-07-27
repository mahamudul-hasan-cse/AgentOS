"use client";

import { useState } from "react";

import { fetchMemoryState, MemoryPage } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";
import { HistoryBadge, useTimeTravel } from "./TimeTravelContext";

function PageCard({ page, tone }: { page: MemoryPage; tone: "ram" | "swap" }) {
  // A copy-on-write SHARED page is styled distinctly from a private one: it
  // occupies no memory of its own, so it reads as "borrowed" rather than owned.
  const shared = Boolean(page.shared);
  const border = shared
    ? "border-violet-500/50"
    : tone === "ram"
      ? "border-emerald-500/30"
      : "border-slate-600/40";
  const dot = shared
    ? "bg-violet-400"
    : tone === "ram"
      ? "bg-emerald-400"
      : "bg-slate-500";
  return (
    <div
      className={`rounded border ${border} px-2.5 py-1.5 ${
        shared ? "bg-violet-500/10" : "bg-black/20"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 font-semibold text-slate-200">
          <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
          {page.page_id}
        </span>
        <span className="flex items-center gap-1.5 shrink-0">
          {shared && (
            <span
              className="rounded border border-violet-500/50 px-1 text-[9px] text-violet-300"
              title="copy-on-write: shared with another agent, copied on first write"
            >
              shared &times;{page.refcount ?? 2}
            </span>
          )}
          <span className="text-xs tabular-nums text-slate-500">
            {page.token_count}t
          </span>
        </span>
      </div>
      {page.content && (
        <p className="mt-0.5 truncate text-xs text-slate-500">{page.content}</p>
      )}
    </div>
  );
}

function CowMetrics({
  cow,
  global_,
}: {
  cow?: Record<string, number>;
  global_?: Record<string, number>;
}) {
  if (!cow && !global_) return null;
  const saved = global_?.tokens_saved ?? 0;
  const ratio = global_?.savings_ratio ?? 0;
  return (
    <div className="mb-3 rounded border border-slate-800 bg-black/20 px-2.5 py-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
        copy-on-write
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        <span className="text-violet-300">
          shared <span className="tabular-nums">{cow?.pages_shared ?? 0}</span>
        </span>
        <span className="text-emerald-300">
          private <span className="tabular-nums">{cow?.pages_private ?? 0}</span>
        </span>
        <span className="text-slate-400">
          COW faults <span className="tabular-nums">{cow?.cow_faults ?? 0}</span>
        </span>
      </div>
      {saved > 0 && (
        <div className="mt-1 text-[10px] text-slate-500">
          kernel-wide: <span className="tabular-nums text-slate-300">{saved}</span> tokens
          saved vs a naive copy-on-fork ({(ratio * 100).toFixed(0)}% less memory,{" "}
          {global_?.frames ?? 0} frames backing {global_?.page_table_entries ?? 0} page-table
          entries)
        </div>
      )}
    </div>
  );
}

export function MemoryView() {
  const [agentId, setAgentId] = useState("demo");
  const { data, error } = usePolling(
    () => fetchMemoryState(agentId),
    2000,
    agentId
  );
  const { isLive, snapshot } = useTimeTravel();

  // in history mode, pull this agent's memory out of the snapshot
  const historical = snapshot?.memory?.[agentId];
  const ramPages: MemoryPage[] = isLive
    ? data?.ram_pages ?? []
    : historical?.ram_pages ?? [];
  const swappedPages: MemoryPage[] = isLive
    ? data?.swapped_pages ?? []
    : historical?.swapped_pages ?? [];
  const tokensUsed = isLive ? data?.ram_tokens_used : historical?.ram_tokens_used;
  const tokenBudget = isLive ? data?.ram_budget_tokens : historical?.ram_budget_tokens;
  const hasData = isLive ? Boolean(data) : Boolean(historical);

  return (
    <Panel
      title="Memory View"
      subtitle={
        tokensUsed !== undefined && tokensUsed !== null
          ? `${tokensUsed} / ${tokenBudget ?? "?"} tokens`
          : "—"
      }
    >
      {!isLive && <HistoryBadge label={snapshot?.label} />}

      {isLive && <CowMetrics cow={data?.cow} global_={data?.cow_global} />}

      <div className="mb-3 flex items-center gap-2">
        <label className="text-xs text-slate-500">agent_id</label>
        <input
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          className="w-40 rounded border border-slate-700 bg-black/30 px-2 py-1 text-sm text-slate-200 outline-none focus:border-sky-500"
          placeholder="agent_id"
        />
      </div>

      {isLive && error && <p className="text-xs text-rose-400">error: {error}</p>}
      {!isLive && !historical && (
        <p className="text-xs text-slate-500">
          agent &apos;{agentId}&apos; had no memory at this point in history
        </p>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <h3 className="mb-2 text-xs uppercase tracking-wider text-emerald-400">
            In RAM ({ramPages.length})
          </h3>
          <div className="space-y-1.5">
            {ramPages.map((p) => (
              <PageCard key={p.page_id} page={p} tone="ram" />
            ))}
            {hasData && ramPages.length === 0 && (
              <p className="text-xs text-slate-600">empty</p>
            )}
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-xs uppercase tracking-wider text-slate-400">
            Swapped → ChromaDB ({swappedPages.length})
          </h3>
          <div className="space-y-1.5">
            {swappedPages.map((p) => (
              <PageCard key={p.page_id} page={p} tone="swap" />
            ))}
            {hasData && swappedPages.length === 0 && (
              <p className="text-xs text-slate-600">empty</p>
            )}
          </div>
        </div>
      </div>
    </Panel>
  );
}
