import type { ReactNode } from "react";
import { Card, Callout, Tag } from "../ui";
import { RunSummaryCard } from "./RunSummaryCard";
import { VramSummary } from "./VramSummary";
import { RationaleRow } from "./RationaleRow";
import { PeftTierBadge } from "./PeftTierBadge";
import { ParallelismCard } from "./ParallelismCard";
import { ModelCandidateList } from "./ModelCandidateList";
import { DataFormatSpec } from "./DataFormatSpec";
import { StrategyConfigDisclosure } from "./StrategyConfigDisclosure";
import type { Recommendation } from "../../types/api";

export interface RationaleReportProps {
  recommendation: Recommendation;
  selectedModelId?: string;
  onSelectModel?: (id: string) => void;
  compact?: boolean;
}

const TOKEN_RE = /(P_max|full_finetune|lora_bf16|qlora_8bit|qlora_4bit|\d+-bit)/g;
const TOKEN_EXACT = /^(P_max|full_finetune|lora_bf16|qlora_8bit|qlora_4bit|\d+-bit)$/;

function withCodeTokens(text: string): ReactNode[] {
  return text.split(TOKEN_RE).map((chunk, i) =>
    TOKEN_EXACT.test(chunk) ? (
      <code key={i} className="mono">
        {chunk}
      </code>
    ) : (
      <span key={i}>{chunk}</span>
    ),
  );
}

export function RationaleReport({
  recommendation,
  selectedModelId,
  onSelectModel,
  compact,
}: RationaleReportProps) {
  const { rationale } = recommendation;
  return (
    <div className="stack" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <RunSummaryCard rec={recommendation} />

      <VramSummary text={rationale.vram_summary} />

      <RationaleRow label="Use case → task" mono>
        {rationale.use_case_resolution}
      </RationaleRow>

      <Card style={{ borderLeft: "3px solid var(--accent)" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            marginBottom: "var(--space-2)",
          }}
        >
          <h4 style={{ margin: 0, color: "var(--text-1)" }}>PEFT decision</h4>
          <PeftTierBadge approach={recommendation.peft_approach} />
        </div>
        <p style={{ margin: 0, color: "var(--text-2)", lineHeight: "var(--leading-normal)" }}>
          {withCodeTokens(rationale.peft_decision)}
        </p>
      </Card>

      <ParallelismCard
        strategy={recommendation.parallelism_strategy}
        decision={rationale.parallelism_decision}
      />

      <RationaleRow label="Model filter">
        <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--space-2)", flexWrap: "wrap" }}>
          {rationale.model_filter}
          {/relax/i.test(rationale.model_filter) && <Tag tone="warning">relaxed retry</Tag>}
        </span>
      </RationaleRow>

      {rationale.warnings.length > 0 && (
        <Callout tone="warning" title="Warnings">
          <ul style={{ margin: 0, paddingLeft: "var(--space-4)" }}>
            {rationale.warnings.map((w, i) => (
              <li key={i}>⚠ {w}</li>
            ))}
          </ul>
        </Callout>
      )}

      <div>
        <h3 style={{ color: "var(--text-1)" }}>Candidate models</h3>
        <ModelCandidateList
          models={recommendation.recommended_models}
          selectedId={selectedModelId}
          onSelect={onSelectModel}
        />
      </div>

      {!compact && (
        <>
          <div>
            <h3 style={{ color: "var(--text-1)" }}>Data format</h3>
            <DataFormatSpec spec={recommendation.data_format} />
          </div>
          <StrategyConfigDisclosure
            strategyConfig={recommendation.strategy_config}
            peftConfig={recommendation.peft_config}
          />
        </>
      )}
    </div>
  );
}
