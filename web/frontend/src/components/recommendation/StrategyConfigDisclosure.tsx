import { CodeBlock } from "../ui";

export interface StrategyConfigDisclosureProps {
  strategyConfig: Record<string, unknown>;
  peftConfig: Record<string, unknown> | null;
}

export function StrategyConfigDisclosure({
  strategyConfig,
  peftConfig,
}: StrategyConfigDisclosureProps) {
  return (
    <details>
      <summary style={{ cursor: "pointer", color: "var(--text-2)" }}>
        Full strategy &amp; PEFT config
      </summary>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "var(--space-4)",
          marginTop: "var(--space-3)",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "var(--step--1)",
              textTransform: "uppercase",
              letterSpacing: "var(--tracking-wide)",
              color: "var(--text-3)",
              marginBottom: "var(--space-2)",
            }}
          >
            Strategy config
          </div>
          <CodeBlock code={JSON.stringify(strategyConfig, null, 2)} language="json" />
        </div>
        <div>
          <div
            style={{
              fontSize: "var(--step--1)",
              textTransform: "uppercase",
              letterSpacing: "var(--tracking-wide)",
              color: "var(--text-3)",
              marginBottom: "var(--space-2)",
            }}
          >
            PEFT config
          </div>
          {peftConfig === null ? (
            <p style={{ margin: 0, color: "var(--text-2)" }}>
              Full fine-tune — no PEFT adapter.
            </p>
          ) : (
            <CodeBlock code={JSON.stringify(peftConfig, null, 2)} language="json" />
          )}
        </div>
      </div>
    </details>
  );
}
