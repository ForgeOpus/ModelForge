import { Card, Badge } from "../ui";
import { parallelismLabel } from "../../lib/format";

export interface ParallelismCardProps {
  strategy: string;
  decision: string;
}

export function ParallelismCard({ strategy, decision }: ParallelismCardProps) {
  if (!decision || !decision.trim()) return null;
  return (
    <Card style={{ borderLeft: "3px solid var(--accent)" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-2)",
          marginBottom: "var(--space-2)",
        }}
      >
        <h4 style={{ margin: 0, color: "var(--text-1)" }}>Parallelism</h4>
        <Badge>{parallelismLabel(strategy)}</Badge>
      </div>
      <p style={{ margin: 0, color: "var(--text-2)", lineHeight: "var(--leading-normal)" }}>
        {decision}
      </p>
    </Card>
  );
}
