import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, ApiError } from "./api";
import type { HardwareSpecs, RunRecord, SystemInfo } from "@/types";

interface Store {
  system: SystemInfo | null;
  hardware: HardwareSpecs | null;
  runs: RunRecord[];
  /** max_active_jobs is 1, so there is at most one of these at any moment. */
  activeRun: RunRecord | null;
  loading: boolean;
  error: string | null;
  /** True when the server rejected our token, which needs a different fix. */
  unauthorised: boolean;
  refreshRuns: () => Promise<void>;
  reload: () => Promise<void>;
}

const StoreContext = createContext<Store>({
  system: null,
  hardware: null,
  runs: [],
  activeRun: null,
  loading: true,
  error: null,
  unauthorised: false,
  refreshRuns: async () => {},
  reload: async () => {},
});

export function StoreProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [hardware, setHardware] = useState<HardwareSpecs | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [unauthorised, setUnauthorised] = useState(false);

  const refreshRuns = useCallback(async () => {
    try {
      setRuns(await api.listRuns(100));
    } catch {
      /* a failed refresh shouldn't blank the list already on screen */
    }
  }, []);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Hardware detection can fail independently of everything else; don't let
      // it take the whole app down.
      const [sys, hw, runList] = await Promise.allSettled([
        api.system(),
        api.hardware(),
        api.listRuns(100),
      ]);
      if (sys.status === "fulfilled") setSystem(sys.value);
      else throw sys.reason;
      if (hw.status === "fulfilled") setHardware(hw.value);
      if (runList.status === "fulfilled") setRuns(runList.value);
      setUnauthorised(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.friendly : String(err));
      setUnauthorised(err instanceof ApiError && (err.status === 401 || err.status === 403));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Keep the global run pill honest without hammering the server.
  useEffect(() => {
    const id = setInterval(() => void refreshRuns(), 8000);
    return () => clearInterval(id);
  }, [refreshRuns]);

  const activeRun = useMemo(
    () => runs.find((r) => r.status === "running" || r.status === "queued") ?? null,
    [runs],
  );

  const value = useMemo(
    () => ({ system, hardware, runs, activeRun, loading, error, unauthorised, refreshRuns, reload }),
    [system, hardware, runs, activeRun, loading, error, unauthorised, refreshRuns, reload],
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export const useStore = () => useContext(StoreContext);
