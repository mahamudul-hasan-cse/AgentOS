"use client";

import { useState } from "react";

import { fetchMemoryState, MemoryPage } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";
import { HistoryBadge, useTimeTravel } from "./TimeTravelContext";

function PageCard({ page, tone }: { page: MemoryPage; tone: "ram" | "swap" }) {
  const border = tone === "ram" ? "border-emerald-500/30" : "border-slate-600/40";
  const dot = tone === "ram" ? "bg-emerald-400" : "bg-slate-500";
  return (
    <div className={`rounded border ${border} bg-black/20 px-2.5 py-1.5`}>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 font-semibold text-slate-200">
          <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
          {page.page_id}
        </span>
        <span className="text-xs tabular-nums text-slate-500">
          {page.token_count}t
        </span>
      </div>
      {page.content && (
        <p className="mt-0.5 truncate text-xs text-slate-500">{page.content}</p>
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
