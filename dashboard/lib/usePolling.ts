"use client";

import { useEffect, useRef, useState } from "react";

interface PollingResult<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Polls `fetcher` every `intervalMs`, re-subscribing when `key` changes. */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  key: unknown = null
): PollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    let active = true;

    const tick = async () => {
      try {
        const result = await fetcherRef.current();
        if (active) {
          setData(result);
          setError(null);
        }
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setLoading(false);
      }
    };

    setLoading(true);
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [intervalMs, key]);

  return { data, error, loading };
}
