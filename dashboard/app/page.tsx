import { GanttChart } from "@/components/GanttChart";
import { MemoryView } from "@/components/MemoryView";
import { ProcessTable } from "@/components/ProcessTable";
import { SyscallTrace } from "@/components/SyscallTrace";

export default function Home() {
  return (
    <main className="mx-auto min-h-screen max-w-7xl px-6 py-6">
      <header className="mb-6">
        <h1 className="text-xl font-bold tracking-tight text-slate-100">
          AgentOS-Lite
          <span className="ml-2 text-sm font-normal text-slate-500">
            kernel dashboard
          </span>
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          live view of the scheduler, paged memory, and syscall trace · polling{" "}
          <span className="text-slate-400">http://localhost:8000</span>
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ProcessTable />
        <GanttChart />
        <MemoryView />
        <SyscallTrace />
      </div>
    </main>
  );
}
