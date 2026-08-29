import { useEffect, useRef, useState } from "react";
import { Card, Badge, Checkbox } from "../ui";
import { isHeartbeat } from "../../types/events";
import type { TrainingEvent } from "../../types/events";
import { RunEventLine } from "./RunEventLine";

export interface RunConsoleProps {
  events: TrainingEvent[];
  connectionStatus: string;
}

function statusColor(status: string): string {
  if (status === "open") return "var(--success)";
  if (status === "error") return "var(--danger)";
  return "var(--text-3)";
}

export function RunConsole({ events, connectionStatus }: RunConsoleProps) {
  const [sticky, setSticky] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const visible = events.filter((e) => !isHeartbeat(e));

  useEffect(() => {
    if (sticky && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visible.length, sticky]);

  return (
    <Card>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-3)",
          marginBottom: "var(--space-3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <strong style={{ color: "var(--text-1)" }}>Event stream</strong>
          <Badge color={statusColor(connectionStatus)}>{connectionStatus}</Badge>
        </div>
        <Checkbox
          label="Stick to bottom"
          checked={sticky}
          onChange={(e) => setSticky(e.target.checked)}
        />
      </div>
      <div
        ref={scrollRef}
        className="mono"
        style={{ maxHeight: "460px", overflow: "auto", display: "flex", flexDirection: "column" }}
      >
        {visible.length === 0 ? (
          <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>No events yet.</span>
        ) : (
          visible.map((e, i) => <RunEventLine key={i} event={e} />)
        )}
      </div>
    </Card>
  );
}
