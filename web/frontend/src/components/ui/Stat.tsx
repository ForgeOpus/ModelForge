import type { ReactNode } from "react";

export interface StatProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
}

export function Stat({ label, value, hint }: StatProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
      <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>{label}</span>
      <span
        className="mono"
        style={{
          fontSize: "var(--step-1)",
          color: "var(--text-1)",
          fontWeight: 600,
          overflowWrap: "anywhere",
        }}
      >
        {value}
      </span>
      {hint && <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>{hint}</span>}
    </div>
  );
}

export function StatGrid({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
        gap: "var(--space-4)",
      }}
    >
      {children}
    </div>
  );
}
