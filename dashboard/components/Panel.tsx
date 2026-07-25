export function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex min-h-0 flex-col rounded-lg border border-slate-800 bg-[#0d1220]">
      <header className="flex items-baseline justify-between border-b border-slate-800 px-4 py-2.5">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200">
          {title}
        </h2>
        {subtitle && <span className="text-xs text-slate-500">{subtitle}</span>}
      </header>
      <div className="min-h-0 flex-1 overflow-auto p-4">{children}</div>
    </section>
  );
}

export function StateBadge({ state }: { state: string }) {
  const styles: Record<string, string> = {
    running: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
    ready: "bg-sky-500/15 text-sky-300 border-sky-500/40",
    waiting: "bg-amber-500/15 text-amber-300 border-amber-500/40",
    terminated: "bg-slate-500/15 text-slate-400 border-slate-500/40",
  };
  const cls = styles[state] ?? "bg-slate-500/15 text-slate-400 border-slate-500/40";
  return (
    <span
      className={`inline-block rounded border px-2 py-0.5 text-xs font-medium ${cls}`}
    >
      {state}
    </span>
  );
}
