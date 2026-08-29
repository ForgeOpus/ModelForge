import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

interface Options {
  enabled?: boolean;
  pollMs?: number;
}

export interface UseApiResult<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new ApiError(0, message);
}

export function useApi<T>(
  fn: () => Promise<T>,
  deps: unknown[],
  opts: Options = {},
): UseApiResult<T> {
  const { enabled = true, pollMs } = opts;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);

  const genRef = useRef(0);
  const didLoadRef = useRef(false);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const run = useCallback((isPoll: boolean) => {
    const gen = ++genRef.current;
    if (!isPoll && !didLoadRef.current) setLoading(true);
    fnRef.current()
      .then((result) => {
        if (gen !== genRef.current) return;
        setData(result);
        setError(null);
      })
      .catch((err: unknown) => {
        if (gen !== genRef.current) return;
        setError(toApiError(err));
      })
      .finally(() => {
        if (gen !== genRef.current) return;
        didLoadRef.current = true;
        setLoading(false);
      });
  }, []);

  const reload = useCallback(() => {
    if (!enabled) return;
    run(false);
  }, [enabled, run]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    run(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);

  useEffect(() => {
    if (!enabled || !pollMs) return;
    const id = setInterval(() => run(true), pollMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, pollMs, ...deps]);

  return { data, error, loading, reload };
}
