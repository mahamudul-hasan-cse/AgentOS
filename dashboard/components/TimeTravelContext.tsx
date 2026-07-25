"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

import { fetchSnapshot, fetchTimeline, Snapshot, SnapshotSummary } from "@/lib/replay";

interface TimeTravelValue {
  timeline: SnapshotSummary[];
  /** null => viewing live state */
  selectedId: number | null;
  snapshot: Snapshot | null;
  isLive: boolean;
  selectSnapshot: (id: number | null) => void;
  returnToLive: () => void;
}

const TimeTravelContext = createContext<TimeTravelValue>({
  timeline: [],
  selectedId: null,
  snapshot: null,
  isLive: true,
  selectSnapshot: () => {},
  returnToLive: () => {},
});

export const useTimeTravel = () => useContext(TimeTravelContext);

export function TimeTravelProvider({ children }: { children: React.ReactNode }) {
  const [timeline, setTimeline] = useState<SnapshotSummary[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);

  // keep the timeline fresh while live; freeze polling once scrubbed into the
  // past so the scrubber doesn't shift under the user's cursor
  useEffect(() => {
    let active = true;
    const tick = async () => {
      try {
        const snapshots = await fetchTimeline();
        if (active) setTimeline(snapshots);
      } catch {
        /* backend down — panels show their own error state */
      }
    };
    tick();
    if (selectedId !== null) return () => { active = false; };
    const id = setInterval(tick, 2000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [selectedId]);

  useEffect(() => {
    if (selectedId === null) {
      setSnapshot(null);
      return;
    }
    let active = true;
    fetchSnapshot(selectedId)
      .then((s) => {
        if (active) setSnapshot(s);
      })
      .catch(() => {
        if (active) setSnapshot(null);
      });
    return () => {
      active = false;
    };
  }, [selectedId]);

  const selectSnapshot = useCallback((id: number | null) => setSelectedId(id), []);
  const returnToLive = useCallback(() => setSelectedId(null), []);

  return (
    <TimeTravelContext.Provider
      value={{
        timeline,
        selectedId,
        snapshot,
        isLive: selectedId === null,
        selectSnapshot,
        returnToLive,
      }}
    >
      {children}
    </TimeTravelContext.Provider>
  );
}

/** Amber banner shown on any panel currently rendering historical state. */
export function HistoryBadge({ label }: { label?: string }) {
  const { returnToLive } = useTimeTravel();
  return (
    <div className="mb-3 flex items-center justify-between gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5">
      <span className="text-xs text-amber-300">
        viewing history{label ? ` — ${label}` : ""}
      </span>
      <button
        onClick={returnToLive}
        className="shrink-0 rounded border border-amber-500/50 px-2 py-0.5 text-xs text-amber-200 hover:bg-amber-500/20"
      >
        return to live
      </button>
    </div>
  );
}
