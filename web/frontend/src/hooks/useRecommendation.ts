import { useCallback, useState } from "react";
import { ApiError, api } from "../api/client";
import type { Recommendation, RecommendationRequest } from "../types/api";

export function useRecommendation(): {
  recommendation: Recommendation | null;
  loading: boolean;
  error: ApiError | null;
  generate: (body: RecommendationRequest) => Promise<Recommendation | null>;
  reset: () => void;
} {
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const generate = useCallback(async (body: RecommendationRequest) => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.recommend(body);
      setRecommendation(result);
      return result;
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err
          : new ApiError(0, err instanceof Error ? err.message : String(err)),
      );
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setRecommendation(null);
    setError(null);
    setLoading(false);
  }, []);

  return { recommendation, loading, error, generate, reset };
}
