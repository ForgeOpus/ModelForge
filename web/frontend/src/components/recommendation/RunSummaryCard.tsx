import { Card, Badge, Tag, Stat, StatGrid } from "../ui";
import { parallelismLabel } from "../../lib/format";
import { PeftTierBadge } from "./PeftTierBadge";
import type { Recommendation } from "../../types/api";

export interface RunSummaryCardProps {
  rec: Recommendation;
}

export function RunSummaryCard({ rec }: RunSummaryCardProps) {
  return (
    <Card>
      <StatGrid>
        <Stat label="Strategy" value={<Tag>{rec.strategy_slug}</Tag>} />
        <Stat label="PEFT" value={<PeftTierBadge approach={rec.peft_approach} />} />
        <Stat label="Task" value={<Tag>{rec.task_slug}</Tag>} />
        <Stat label="Provider" value={rec.provider_slug} />
        <Stat
          label="Parallelism"
          value={<Badge>{parallelismLabel(rec.parallelism_strategy)}</Badge>}
        />
      </StatGrid>
    </Card>
  );
}
