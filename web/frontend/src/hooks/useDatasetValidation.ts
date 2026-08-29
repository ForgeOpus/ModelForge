import { useCallback, useState } from "react";
import { ApiError, api } from "../api/client";
import type { DatasetValidationRequest, DatasetValidationResult } from "../types/api";

export function useDatasetValidation(): {
  result: DatasetValidationResult | null;
  loading: boolean;
  error: ApiError | null;
  validate: (body: DatasetValidationRequest) => Promise<void>;
  reset: () => void;
} {
  const [result, setResult] = useState<DatasetValidationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const validate = useCallback(async (body: DatasetValidationRequest) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.validateDataset(body);
      setResult(res);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err
          : new ApiError(0, err instanceof Error ? err.message : String(err)),
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, validate, reset };
}
