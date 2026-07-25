"use client";

import { useState } from "react";

import { fetchMemoryState, MemoryPage } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";

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
      <p className="mt-0.5 truncate text-xs text-slate-500">{page.content}</p>
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

  return (
    <Panel
      title="Memory View"
      subtitle={
        data ? `${data.ram_tokens_used} / ${data.ram_budget_tokens} tokens` : "—"
      }
    >
      <div className="mb-3 flex items-center gap-2">
        <label className="text-xs text-slate-500">agent_id</label>
        <input
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          className="w-40 rounded border border-slate-700 bg-black/30 px-2 py-1 text-sm text-slate-200 outline-none focus:border-sky-500"
          placeholder="agent_id"
        />
      </div>

      {error && <p className="text-xs text-rose-400">error: {error}</p>}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <h3 className="mb-2 text-xs uppercase tracking-wider text-emerald-400">
            In RAM ({data?.ram_pages.length ?? 0})
          </h3>
          <div className="space-y-1.5">
            {(data?.ram_pages ?? []).map((p) => (
              <PageCard key={p.page_id} page={p} tone="ram" />
            ))}
            {data && data.ram_pages.length === 0 && (
              <p className="text-xs text-slate-600">empty</p>
            )}
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-xs uppercase tracking-wider text-slate-400">
            Swapped → ChromaDB ({data?.swapped_pages.length ?? 0})
          </h3>
          <div className="space-y-1.5">
            {(data?.swapped_pages ?? []).map((p) => (
              <PageCard key={p.page_id} page={p} tone="swap" />
            ))}
            {data && data.swapped_pages.length === 0 && (
              <p className="text-xs text-slate-600">empty</p>
            )}
          </div>
        </div>
      </div>
    </Panel>
  );
}
