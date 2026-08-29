import { eventMeta } from "../../types/events";
import type { TrainingEvent } from "../../types/events";

export interface RunEventLineProps {
  event: TrainingEvent;
}

export function RunEventLine({ event }: RunEventLineProps) {
  if (event.type === "heartbeat") return null;

  const meta = eventMeta(event.type);

  let suffix: string | null = null;
  let path: string | null = null;
  if (event.type === "log") {
    const { loss, epoch } = event.payload.logs;
    const bits: string[] = [];
    if (typeof loss === "number") bits.push(`loss ${loss.toFixed(4)}`);
    if (typeof epoch === "number") bits.push(`epoch ${epoch.toFixed(2)}`);
    if (bits.length) suffix = bits.join("  ");
  } else if (event.type === "checkpoint") {
    path = event.payload.path;
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: "var(--space-2)",
        fontSize: "var(--step--1)",
        padding: "2px 0",
      }}
    >
      <span
        aria-hidden
        style={{
          width: "6px",
          height: "6px",
          borderRadius: "var(--radius-pill)",
          background: meta.color,
          flexShrink: 0,
          transform: "translateY(1px)",
        }}
      />
      <span className="mono" style={{ color: meta.color, flexShrink: 0 }}>
        {meta.label}
      </span>
      <span style={{ color: "var(--text-2)", flex: 1, minWidth: 0 }}>
        {event.message}
        {suffix && <span style={{ color: "var(--text-3)" }}>  {suffix}</span>}
      </span>
      {path && (
        <span
          className="mono"
          title={path}
          style={{
            color: "var(--text-3)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            maxWidth: "50%",
          }}
        >
          {path}
        </span>
      )}
    </div>
  );
}
