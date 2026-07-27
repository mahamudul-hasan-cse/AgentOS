"use client";

import { API_BASE } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Panel } from "./Panel";

export interface TreeNode {
  pid: string;
  state: string;
  parent_pid: string | null;
  priority: number;
  remaining_burst: number;
  exit_status: number | null;
  children: TreeNode[];
}

async function fetchTree(): Promise<TreeNode> {
  const res = await fetch(`${API_BASE}/scheduler/tree`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/scheduler/tree -> ${res.status}`);
  return res.json();
}

const STATE_STYLES: Record<string, string> = {
  running: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  ready: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  waiting: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  terminated: "bg-slate-500/15 text-slate-400 border-slate-500/40",
  // zombies get their own alarming treatment so they stand out at a glance
  zombie: "bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/50",
};

function countZombies(node: TreeNode): number {
  return (
    (node.state === "zombie" ? 1 : 0) +
    node.children.reduce((sum, child) => sum + countZombies(child), 0)
  );
}

function Node({ node, depth = 0, isLast = true, prefix = "" }: {
  node: TreeNode;
  depth?: number;
  isLast?: boolean;
  prefix?: string;
}) {
  const isZombie = node.state === "zombie";
  const badge = STATE_STYLES[node.state] ?? STATE_STYLES.terminated;
  const branch = depth === 0 ? "" : isLast ? "└─ " : "├─ ";
  const childPrefix = depth === 0 ? "" : prefix + (isLast ? "   " : "│  ");

  return (
    <div>
      <div className="flex items-center gap-2 whitespace-pre font-mono text-xs leading-6">
        <span className="text-slate-600">{prefix + branch}</span>
        <span
          className={`font-semibold ${
            isZombie ? "text-fuchsia-300 line-through decoration-fuchsia-500/50" : "text-slate-100"
          }`}
        >
          {node.pid}
        </span>
        <span className={`rounded border px-1.5 py-px text-[10px] ${badge}`}>
          {isZombie ? "zombie <defunct>" : node.state}
        </span>
        {isZombie && node.exit_status !== null && (
          <span className="text-[10px] text-fuchsia-400/80">
            exit={node.exit_status}
          </span>
        )}
      </div>
      {node.children.map((child, i) => (
        <Node
          key={child.pid}
          node={child}
          depth={depth + 1}
          isLast={i === node.children.length - 1}
          prefix={childPrefix}
        />
      ))}
    </div>
  );
}

export function ProcessTree() {
  const { data, error } = usePolling(fetchTree, 2000);
  const zombies = data ? countZombies(data) : 0;

  return (
    <Panel
      title="Process Tree"
      subtitle={zombies > 0 ? `${zombies} zombie(s) awaiting reap` : "hierarchy"}
    >
      {error && <p className="text-xs text-rose-400">backend unreachable: {error}</p>}
      {!error && !data && <p className="text-xs text-slate-500">loading…</p>}
      {data && (
        <>
          <div className="overflow-x-auto">
            <Node node={data} />
          </div>
          {zombies > 0 && (
            <p className="mt-3 text-[10px] text-fuchsia-400/80">
              zombies hold their exit status until the parent reaps them (WAIT)
            </p>
          )}
        </>
      )}
    </Panel>
  );
}
