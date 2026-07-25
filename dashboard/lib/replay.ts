import { API_BASE, MemoryPage, ProcessRow } from "./api";

export interface SnapshotSummary {
  snapshot_id: number;
  timestamp: number;
  syscall_id: string | null;
  label: string;
}

export interface SnapshotMemory {
  ram_pages: MemoryPage[];
  swapped_pages: MemoryPage[];
  ram_tokens_used: number;
  ram_budget_tokens: number | null;
}

export interface Snapshot {
  snapshot_id: number;
  timestamp: number;
  syscall_id: string | null;
  label: string;
  processes: ProcessRow[];
  memory: Record<string, SnapshotMemory>;
  resources: Record<string, Record<string, unknown>>;
  quotas: Record<string, Record<string, number>>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const fetchTimeline = () =>
  getJSON<{ snapshots: SnapshotSummary[] }>("/replay/timeline").then(
    (r) => r.snapshots
  );

export const fetchSnapshot = (id: number) =>
  getJSON<Snapshot>(`/replay/snapshot/${id}`);
