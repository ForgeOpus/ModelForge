import { api } from "../api/client";
import { useApi } from "./useApi";

export function useRun(runId: number | null) {
  return useApi(() => api.getRun(runId as number), [runId], {
    enabled: runId != null,
  });
}
