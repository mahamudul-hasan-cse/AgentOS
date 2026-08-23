export function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`flex h-full min-h-0 flex-col rounded-lg border border-slate-800 bg-[#0d1220] shadow-sm shadow-black/20 ${className}`}
    >
      <header className="flex shrink-0 items-baseline justify-between gap-3 border-b border-slate-800 px-4 py-2.5">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200">
          {title}
        </h2>
        {subtitle && (
          <span className="truncate text-xs text-slate-500">{subtitle}</span>
        )}
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
    success: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
    failed: "bg-rose-500/15 text-rose-300 border-rose-500/40",
    pending: "bg-slate-500/15 text-slate-400 border-slate-500/40",
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
