/*
 * WebSocket events from WS /api/runs/{id}/events.
 * Emitted by workflow/jobs.py: log | checkpoint | cancel | status | completed | cancelled | error.
 * app.py injects "heartbeat" roughly every second while idle — the UI ignores it.
 */

export interface EventBase {
  run_id: number;
  type: string;
  message: string;
  payload: Record<string, unknown>;
}

export interface LogEvent extends EventBase {
  type: "log";
  payload: {
    step: number;
    max_steps: number;
    logs: { loss?: number; epoch?: number; learning_rate?: number; [k: string]: unknown };
  };
}

export interface CheckpointEvent extends EventBase {
  type: "checkpoint";
  payload: { path: string; step: number };
}

export interface StatusEvent extends EventBase {
  type: "status" | "cancel";
}

export interface TerminalEvent extends EventBase {
  type: "completed" | "cancelled" | "error" | "failed";
  payload: { output_dir?: string; error?: string };
}

export interface HeartbeatEvent {
  run_id: number;
  type: "heartbeat";
}

export type TrainingEvent =
  | LogEvent
  | CheckpointEvent
  | StatusEvent
  | TerminalEvent
  | HeartbeatEvent;

const TERMINAL = new Set(["completed", "cancelled", "error", "failed"]);

export function isTerminal(type: string): boolean {
  return TERMINAL.has(type);
}

export function isHeartbeat(ev: { type: string }): ev is HeartbeatEvent {
  return ev.type === "heartbeat";
}

export function isLogEvent(ev: TrainingEvent): ev is LogEvent {
  return ev.type === "log";
}

/** Maps an event type to a semantic colour token + short label. */
export function eventMeta(type: string): { color: string; label: string } {
  switch (type) {
    case "status":
      return { color: "var(--text-3)", label: "status" };
    case "log":
      return { color: "var(--text-3)", label: "log" };
    case "checkpoint":
      return { color: "var(--peft-lora)", label: "checkpoint" };
    case "cancel":
    case "cancelled":
      return { color: "var(--warning)", label: type };
    case "completed":
      return { color: "var(--success)", label: "completed" };
    case "error":
    case "failed":
      return { color: "var(--danger)", label: type };
    default:
      return { color: "var(--text-3)", label: type };
  }
}
