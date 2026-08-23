"use client";

import { API_BASE } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useMemo, useState } from "react";
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
  zombie: "bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/50",
};

function countZombies(node: TreeNode): number {
  return (
    (node.state === "zombie" ? 1 : 0) +
    node.children.reduce((sum, child) => sum + countZombies(child), 0)
  );
}

function countTerminated(node: TreeNode): number {
  return (
    (node.state === "terminated" ? 1 : 0) +
    node.children.reduce((sum, child) => sum + countTerminated(child), 0)
  );
}

/** Drop terminated leaves/subtrees; keep zombies and any node with live descendants. */
function pruneTerminated(node: TreeNode): TreeNode | null {
  const children = node.children
    .map(pruneTerminated)
    .filter((c): c is TreeNode => c != null);

  if (node.state === "zombie") {
    return { ...node, children };
  }
  if (node.state === "terminated" && children.length === 0) {
    return null;
  }
  return { ...node, children };
}

function Node({
  node,
  depth = 0,
  isLast = true,
  prefix = "",
}: {
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
            isZombie
              ? "text-fuchsia-300 line-through decoration-fuchsia-500/50"
              : "text-slate-100"
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
  const [showTerminated, setShowTerminated] = useState(false);

  const zombies = data ? countZombies(data) : 0;
  const terminated = data ? countTerminated(data) : 0;

  const display = useMemo(() => {
    if (!data) return null;
    if (showTerminated) return data;
    return pruneTerminated(data) ?? data;
  }, [data, showTerminated]);

  return (
    <Panel
      title="Process Tree"
      subtitle={
        zombies > 0 ? `${zombies} zombie(s) awaiting reap` : "hierarchy"
      }
    >
      {error && (
        <p className="text-xs text-rose-400">backend unreachable: {error}</p>
      )}
      {!error && !data && <p className="text-xs text-slate-500">loading…</p>}
      {display && (
        <>
          <div className="max-h-[280px] overflow-auto">
            <Node node={display} />
          </div>
          {zombies > 0 && (
            <p className="mt-3 text-[10px] text-fuchsia-400/80">
              zombies hold their exit status until the parent reaps them (WAIT)
            </p>
          )}
          {terminated > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-800/80 pt-2 text-[11px] text-slate-500">
              <span>
                {showTerminated
                  ? `showing all · ${terminated} terminated`
                  : `${terminated} terminated collapsed`}
              </span>
              <button
                type="button"
                onClick={() => setShowTerminated((v) => !v)}
                className="rounded border border-slate-700 px-2 py-0.5 text-slate-400 hover:border-slate-500 hover:text-slate-200"
              >
                {showTerminated ? "hide terminated" : "show terminated"}
              </button>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}
