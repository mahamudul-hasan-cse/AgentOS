import { API_BASE } from "@/lib/api";
import { Chat } from "@/components/Chat";
import { Deadlock } from "@/components/Deadlock";
import { MemoryView } from "@/components/MemoryView";
import { Pipeline } from "@/components/Pipeline";
import { ProcessTable } from "@/components/ProcessTable";
import { ProcessTree } from "@/components/ProcessTree";
import { SyscallTrace } from "@/components/SyscallTrace";

function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-8">
      <div className="mb-3 flex items-end justify-between gap-3 border-b border-slate-800/80 pb-2">
        <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
          {title}
        </h2>
        {hint && <p className="text-[11px] text-slate-600">{hint}</p>}
      </div>
      {children}
    </section>
  );
}

export default function Home() {
  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <header className="mb-8 rounded-lg border border-slate-800 bg-[#0d1220] px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-100">
              AgentOS
              <span className="ml-2 text-sm font-normal text-slate-500">
                kernel dashboard
              </span>
            </h1>
            <p className="mt-1 text-xs text-slate-500">
              Live scheduler, memory, syscalls, and kernel-governed workloads
            </p>
          </div>
          <div className="rounded border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-[11px] text-slate-400">
            API{" "}
            <span className="font-medium text-sky-300/90">{API_BASE}</span>
          </div>
        </div>
      </header>

      <Section title="Kernel state" hint="active processes only · terminated collapsed">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:items-stretch">
          <div className="min-h-[180px] max-h-[340px]">
            <ProcessTable />
          </div>
          <div className="min-h-[180px] max-h-[340px]">
            <ProcessTree />
          </div>
        </div>
      </Section>

      <Section title="Observation" hint="memory · syscalls · deadlock">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3 xl:items-stretch">
          <div className="min-h-[240px]">
            <MemoryView />
          </div>
          <div className="min-h-[240px]">
            <SyscallTrace />
          </div>
          <div className="min-h-[240px] md:col-span-2 xl:col-span-1">
            <Deadlock />
          </div>
        </div>
      </Section>

      <Section title="Workloads" hint="pipeline + kernel assistant">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:items-stretch">
          <div className="min-h-[420px]">
            <Pipeline />
          </div>
          <div className="min-h-[420px]">
            <Chat />
          </div>
        </div>
      </Section>
    </main>
  );
}
