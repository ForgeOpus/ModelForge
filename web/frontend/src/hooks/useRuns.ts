import { api } from "../api/client";
import type { RunRecord } from "../types/api";
import { useApi } from "./useApi";

export function useRuns(opts?: { pollMs?: number }) {
  const result = useApi(() => api.listRuns(100), [], { pollMs: opts?.pollMs });
  return { ...result, runs: (result.data?.runs ?? []) as RunRecord[] };
}
