import { API_BASE } from "@/lib/api";
import { Chat } from "@/components/Chat";
import { Deadlock } from "@/components/Deadlock";
import { GanttChart } from "@/components/GanttChart";
import { HealthBadge } from "@/components/HealthBadge";
import { MemoryView } from "@/components/MemoryView";
import { Pipeline } from "@/components/Pipeline";
import { ProcessTable } from "@/components/ProcessTable";
import { ProcessTree } from "@/components/ProcessTree";
import { SyscallTrace } from "@/components/SyscallTrace";
import { TimeTravel } from "@/components/TimeTravel";
import { TimeTravelProvider } from "@/components/TimeTravelContext";

export default function Home() {
  return (
    <main className="mx-auto min-h-screen max-w-7xl px-6 py-6">
      <header className="mb-6">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-bold tracking-tight text-slate-100">
            AgentOS-Lite
            <span className="ml-2 text-sm font-normal text-slate-500">
              kernel dashboard
            </span>
          </h1>
          <HealthBadge />
        </div>
        <p className="mt-1 text-xs text-slate-500">
          live view of the scheduler, paged memory, and syscall trace · polling{" "}
          <span className="text-slate-400">{API_BASE}</span>
        </p>
      </header>

      <TimeTravelProvider>
        <div className="mb-4">
          <TimeTravel />
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ProcessTable />
          <GanttChart />
          <ProcessTree />
          <MemoryView />
          <SyscallTrace />
          <Deadlock />
          <Pipeline />
          <Chat />
        </div>
      </TimeTravelProvider>
    </main>
  );
}
