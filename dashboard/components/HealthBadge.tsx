"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthStatus } from "@/lib/api";

function backendLabel(health: HealthStatus): string {
  if (health.embedding_backend.includes("Ollama")) return "Ollama embeddings";
  if (health.embedding_backend.includes("Hashing")) return "Hashing embeddings";
  return health.embedding_backend;
}

export function HealthBadge() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const data = await fetchHealth();
        if (!active) return;
        setHealth(data);
        setError(null);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "health unavailable");
      }
    };

    load();
    const id = window.setInterval(load, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, []);

  if (error) {
    return (
      <span className="rounded-full border border-rose-500/40 bg-rose-500/10 px-3 py-1 text-xs text-rose-300">
        backend health unavailable
      </span>
    );
  }

  if (!health) {
    return (
      <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs text-slate-400">
        checking backend health
      </span>
    );
  }

  const semantic = health.semantic_embeddings;
  return (
    <span
      title={health.embedding_backend}
      className={
        semantic
          ? "rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300"
          : "rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs text-amber-300"
      }
    >
      {backendLabel(health)}
      {!semantic && " · degraded"}
    </span>
  );
}
