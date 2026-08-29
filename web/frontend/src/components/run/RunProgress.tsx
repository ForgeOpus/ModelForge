import { Badge, ProgressBar } from "../ui";
import { isTerminal } from "../../types/events";
import type { LogEvent } from "../../types/events";

export interface RunProgressProps {
  lastLog: LogEvent | null;
  status: string;
}

export function RunProgress({ lastLog, status }: RunProgressProps) {
  if (!lastLog) {
    if (isTerminal(status)) return null;
    return (
      <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>
        Waiting for the first training step…
      </span>
    );
  }

  const { step, max_steps, logs } = lastLog.payload;
  const lr = typeof logs.learning_rate === "number" ? logs.learning_rate : undefined;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
      <ProgressBar
        value={max_steps > 0 ? step / max_steps : 0}
        label={`Step ${step} / ${max_steps}`}
      />
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)" }} className="cluster">
        {typeof logs.loss === "number" && <Badge>loss {logs.loss.toFixed(4)}</Badge>}
        {typeof logs.epoch === "number" && <Badge>epoch {logs.epoch.toFixed(2)}</Badge>}
        {lr !== undefined && <Badge>lr {lr}</Badge>}
      </div>
    </div>
  );
}
