import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { api, hasToken } from "../api/client";

interface RunActivity {
  activeRunId: number | null;
  setActiveRunId: (id: number | null) => void;
  refreshActivity: () => Promise<void>;
}

const Ctx = createContext<RunActivity>({
  activeRunId: null,
  setActiveRunId: () => {},
  refreshActivity: async () => {},
});

export const useRunActivity = () => useContext(Ctx);

export function RunActivityProvider({ children }: { children: ReactNode }) {
  const [activeRunId, setActiveRunId] = useState<number | null>(null);

  const refreshActivity = useCallback(async () => {
    if (!hasToken()) {
      setActiveRunId(null);
      return;
    }
    try {
      const { runs } = await api.listRuns(1);
      const newest = runs[0];
      if (newest && (newest.status === "running" || newest.status === "queued")) {
        setActiveRunId(newest.id);
      } else {
        setActiveRunId(null);
      }
    } catch {
      /* swallow (e.g. 401) */
    }
  }, []);

  useEffect(() => {
    void refreshActivity();
  }, [refreshActivity]);

  return (
    <Ctx.Provider value={{ activeRunId, setActiveRunId, refreshActivity }}>
      {children}
    </Ctx.Provider>
  );
}
