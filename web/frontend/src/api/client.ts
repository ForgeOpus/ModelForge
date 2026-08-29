import type {
  CheckpointRecord,
  DatasetValidationRequest,
  DatasetValidationResult,
  HardwareSpecs,
  HealthResponse,
  Recommendation,
  RecommendationRequest,
  RunRecord,
  SystemInfo,
  TrainingRunRequest,
  UseCase,
} from "../types/api";

const TOKEN_KEY = "modelforge_token";
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/** Read a ?token= from the URL (persisting it), else fall back to storage. */
export function resolveToken(): string {
  try {
    const q = new URLSearchParams(window.location.search).get("token");
    if (q) localStorage.setItem(TOKEN_KEY, q);
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(value: string): void {
  try {
    if (value) localStorage.setItem(TOKEN_KEY, value);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

export function hasToken(): boolean {
  return resolveToken().length > 0;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "X-ModelForge-Token": resolveToken(),
        ...(init.headers ?? {}),
      },
    });
  } catch (err) {
    throw new ApiError(0, `Cannot reach the ModelForge server. ${(err as Error).message}`);
  }

  const text = await res.text();
  let data: unknown;
  try {
    data = text ? JSON.parse(text) : undefined;
  } catch {
    data = text;
  }

  if (!res.ok) {
    const detail =
      (data && typeof data === "object" && ("detail" in data || "error" in data)
        ? ((data as Record<string, unknown>).detail ?? (data as Record<string, unknown>).error)
        : undefined) ??
      (typeof data === "string" && data ? data : undefined) ??
      res.statusText;
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new ApiError(res.status, message, data);
  }

  return data as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  system: () => request<SystemInfo>("/api/system"),
  hardware: () => request<HardwareSpecs>("/api/hardware"),
  useCases: () => request<{ use_cases: UseCase[] }>("/api/use-cases"),
  recommend: (body: RecommendationRequest) =>
    request<Recommendation>("/api/recommendations", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  validateDataset: (body: DatasetValidationRequest) =>
    request<DatasetValidationResult>("/api/datasets/validate", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listRuns: (limit = 100) =>
    request<{ runs: RunRecord[] }>(`/api/runs?limit=${limit}`),
  getRun: (id: number) => request<RunRecord>(`/api/runs/${id}`),
  createRun: (body: TrainingRunRequest) =>
    request<RunRecord>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (id: number) =>
    request<RunRecord>(`/api/runs/${id}/cancel`, { method: "POST" }),
  checkpoints: (id: number) =>
    request<{ checkpoints: CheckpointRecord[] }>(`/api/runs/${id}/checkpoints`),
  resumeRun: (id: number, checkpoint_path: string) =>
    request<RunRecord>(`/api/runs/${id}/resume`, {
      method: "POST",
      body: JSON.stringify({ checkpoint_path }),
    }),
};
