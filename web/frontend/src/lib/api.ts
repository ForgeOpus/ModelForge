import type {
  CheckpointRecord,
  DatasetValidation,
  HardwareSpecs,
  Recommendation,
  RunRecord,
  SavedModel,
  SystemInfo,
  TrainingEvent,
} from "@/types";

const TOKEN_KEY = "modelforge_token";

let token = "";

/* The launcher opens /app?token=… . Capture it, then strip it from the URL so
   the bearer token doesn't sit in the address bar and browser history for the
   rest of the session. Safe to call again after a client navigation or a
   refresh: a missing query token leaves the stored one in place. */
export function captureStartupToken(): string {
  if (typeof window === "undefined") return token;
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get("token");
  if (fromUrl) {
    localStorage.setItem(TOKEN_KEY, fromUrl);
    params.delete("token");
    const qs = params.toString();
    window.history.replaceState(
      {},
      "",
      window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash,
    );
  }
  token = localStorage.getItem(TOKEN_KEY) ?? "";
  return token;
}

captureStartupToken();

export function getToken(): string {
  return token;
}

export function setToken(value: string) {
  token = value;
  localStorage.setItem(TOKEN_KEY, value);
}

/** An error that carries the server's own explanation rather than a raw body. */
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** Copy written for the person reading it, not for the log. */
  get friendly(): string {
    switch (this.status) {
      case 401:
      case 403:
        return "This session is not authorised. Restart ModelForge and open the link it prints.";
      case 404:
        return this.detail || "That item no longer exists.";
      case 409:
        return this.detail || "A training run is already active. Cancel it before starting another.";
      case 422:
        return this.detail || "Some values are not valid. Check the highlighted fields.";
      default:
        return this.detail || `The server returned ${this.status}.`;
    }
  }
}

/** FastAPI puts the useful message in {detail}, which may itself be a
    validation-error array. Unwrap it rather than dumping raw JSON at people. */
async function readError(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const body = JSON.parse(text);
    const detail = body?.detail ?? body?.message;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: { loc?: (string | number)[]; msg?: string }) => {
          const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
          return field ? `${field}: ${d.msg}` : (d.msg ?? "");
        })
        .filter(Boolean)
        .join("; ");
    }
    if (detail) return JSON.stringify(detail);
  } catch {
    /* not JSON — fall through to the raw text */
  }
  return text.slice(0, 300) || res.statusText;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "X-ModelForge-Token": token,
        ...(init.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the ModelForge server. Is it still running?");
  }
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => request<{ status: string; version: string }>("/api/health"),
  system: () => request<SystemInfo>("/api/system"),
  hardware: () => request<HardwareSpecs>("/api/hardware"),

  recommend: (body: {
    use_case_slug: string;
    objective: string;
    gpu_indices?: number[] | null;
    n?: number;
  }) => post<Recommendation>("/api/recommendations", body),

  validateDataset: (body: {
    path: string;
    remote: boolean;
    use_case_slug?: string;
    objective?: string;
  }) => post<DatasetValidation>("/api/datasets/validate", body),

  listRuns: (limit = 100) =>
    request<{ runs: RunRecord[] }>(`/api/runs?limit=${limit}`).then((r) => r.runs ?? []),
  getRun: (id: number) => request<RunRecord>(`/api/runs/${id}`),
  startRun: (body: Record<string, unknown>) => post<RunRecord>("/api/runs", body),
  cancelRun: (id: number) => post<RunRecord>(`/api/runs/${id}/cancel`),
  checkpoints: (id: number) =>
    request<{ checkpoints: CheckpointRecord[] }>(`/api/runs/${id}/checkpoints`).then(
      (r) => r.checkpoints ?? [],
    ),
  resumeRun: (id: number, checkpoint_path: string) =>
    post<RunRecord>(`/api/runs/${id}/resume`, { checkpoint_path }),
  /** Replay buffer, so the Run screen survives a page reload. */
  runEvents: (id: number) =>
    request<{ events: TrainingEvent[] }>(`/api/runs/${id}/events`).then((r) => r.events ?? []),

  listModels: () =>
    request<{ models: SavedModel[] }>("/api/models").then((r) => r.models ?? []),
  getModel: (id: number) => request<SavedModel>(`/api/models/${id}`),
  deleteModel: (id: number, deleteFiles: boolean) =>
    request<{ deleted: boolean }>(`/api/models/${id}?delete_files=${deleteFiles}`, {
      method: "DELETE",
    }),
  validateModelId: (model_id: string, task_slug?: string) =>
    post<{ valid: boolean; model_id: string; num_parameters: number | null; error: string | null }>(
      "/api/models/validate",
      { model_id, task_slug },
    ),
  pushModel: (id: number, repo_id: string, priv: boolean) =>
    post<{ pushed: boolean; repo_id: string; url: string }>(`/api/models/${id}/push`, {
      repo_id,
      private: priv,
    }),

  playgroundStatus: () =>
    request<{ available: boolean; reason: string | null; session_id: string | null }>(
      "/api/playground/status",
    ),
  openPlayground: (model_path: string) =>
    post<{ session_id: string; model_path: string; base_model: string | null; note?: string | null }>(
      "/api/playground/sessions",
      { model_path },
    ),
  closePlayground: (sessionId: string) =>
    request<{ closed: boolean }>(`/api/playground/sessions/${sessionId}`, { method: "DELETE" }),
};

export function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const sep = path.includes("?") ? "&" : "?";
  return `${proto}://${window.location.host}${path}${sep}token=${encodeURIComponent(token)}`;
}
