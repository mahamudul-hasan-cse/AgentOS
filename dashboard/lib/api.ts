export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface ProcessRow {
  pid: string;
  state: "ready" | "running" | "waiting" | "terminated" | string;
  arrival_time: number;
  estimated_burst: number;
  remaining_burst: number;
  priority: number;
}

export interface GanttSlice {
  pid: string;
  start: number;
  end: number;
}

export interface SchedulerState {
  algorithm: string | null;
  processes: ProcessRow[];
  timeline: GanttSlice[];
}

export interface MemoryPage {
  page_id: string;
  /** omitted by replay snapshots, which store page identity only */
  content?: string;
  token_count: number;
  last_accessed?: number | null;
  /** copy-on-write: true when another agent references the same frame */
  shared?: boolean;
  refcount?: number;
}

export interface MemoryState {
  agent_id: string;
  ram_budget_tokens: number;
  ram_tokens_used: number;
  ram_pages: MemoryPage[];
  swapped_pages: MemoryPage[];
  /** per-agent COW accounting: pages_shared / pages_private / cow_faults */
  cow?: Record<string, number>;
  /** kernel-wide COW accounting incl. tokens_saved vs a naive copy-on-fork */
  cow_global?: Record<string, number>;
}

export interface SyscallEntry {
  syscall_id: string;
  agent_id: string;
  type: string;
  status: string;
  latency_ms: number | null;
  timestamp: number;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const fetchSchedulerState = () =>
  getJSON<SchedulerState>("/scheduler/state");

export const fetchMemoryState = (agentId: string) =>
  getJSON<MemoryState>(`/memory/state/${encodeURIComponent(agentId)}`);

export const fetchSyscallLog = () =>
  getJSON<{ syscalls: SyscallEntry[] }>("/syscalls/log?limit=20").then(
    (r) => r.syscalls
  );

// stable color per pid for the Gantt chart + process rows
const PALETTE = [
  "#89b4fa",
  "#a6e3a1",
  "#f9e2af",
  "#f38ba8",
  "#cba6f7",
  "#94e2d5",
  "#fab387",
  "#74c7ec",
];

export function colorForPid(pid: string): string {
  let hash = 0;
  for (let i = 0; i < pid.length; i++) {
    hash = (hash * 31 + pid.charCodeAt(i)) & 0xffffffff;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

// --- flagship pipeline -------------------------------------------------------

export interface PipelineEvent {
  syscall_id?: string;
  type?: string;
  status?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface PipelineStage {
  stage: string;
  agent_id: string;
  status: string;
  produced: string | null;
  file: string | null;
  driver_used: string | null;
  error: string | null;
  quota_events: PipelineEvent[];
  resource_events: PipelineEvent[];
}

export interface PipelineTester {
  passed: boolean;
  exit_code: number | null;
  stdout: string;
  stderr: string;
  timeout: boolean;
  rejected: boolean;
  duration_ms?: number;
  sandbox?: Record<string, unknown>;
}

export interface PipelineStatus {
  run_id?: string;
  topic?: string;
  coordinator_id?: string;
  status: string;
  current_stage: string | null;
  stages: PipelineStage[];
  final_report: string | null;
  tester: PipelineTester | null;
  events: PipelineEvent[];
  sandbox_review_note?: string;
}

export const fetchPipelineStatus = () =>
  getJSON<PipelineStatus>("/pipeline/status");

export async function runPipeline(topic: string): Promise<PipelineStatus> {
  const res = await fetch(`${API_BASE}/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
