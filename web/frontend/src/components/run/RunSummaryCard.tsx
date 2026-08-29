import { Card, Callout, Stat, StatGrid, Tag } from "../ui";
import { PeftTierBadge } from "../recommendation";
import type { RunRecord } from "../../types/api";

export interface RunSummaryCardProps {
  run: RunRecord;
}

const truncate: React.CSSProperties = {
  maxWidth: "100%",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

export function RunSummaryCard({ run }: RunSummaryCardProps) {
  return (
    <Card>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
        <StatGrid>
          <Stat label="Model" value={<span className="mono">{run.model_id}</span>} />
          <Stat label="Use case" value={run.use_case} />
          <Stat label="Objective" value={run.objective} />
          <Stat label="Task" value={<Tag>{run.task_slug ?? "—"}</Tag>} />
          <Stat label="Strategy" value={<Tag>{run.strategy ?? "—"}</Tag>} />
          <Stat
            label="PEFT"
            value={run.peft ? <PeftTierBadge approach={run.peft} /> : "—"}
          />
          <Stat
            label="Output dir"
            value={
              <span className="mono" title={run.output_dir} style={truncate}>
                {run.output_dir}
              </span>
            }
          />
        </StatGrid>

        {run.error && (
          <Callout tone="danger" title="Run error">
            <span style={{ whiteSpace: "pre-wrap" }}>{run.error}</span>
          </Callout>
        )}
      </div>
    </Card>
  );
}
