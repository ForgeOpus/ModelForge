import { Badge } from "../ui";
import { statusMeta } from "../../lib/format";
import type { RunStatus } from "../../types/api";

export interface RunStatusBadgeProps {
  status: RunStatus | string;
  size?: "sm" | "md";
}

export function RunStatusBadge({ status }: RunStatusBadgeProps) {
  const meta = statusMeta(status);
  return (
    <Badge color={meta.color}>
      {meta.active && (
        <span
          aria-hidden
          style={{
            width: "6px",
            height: "6px",
            borderRadius: "var(--radius-pill)",
            background: "currentColor",
            animation: "activityPulse 2s infinite",
          }}
        />
      )}
      {meta.label}
    </Badge>
  );
}
