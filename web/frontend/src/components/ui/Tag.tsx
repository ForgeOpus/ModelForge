import type { ReactNode } from "react";

export interface TagProps {
  tone?: "neutral" | "warning";
  children: ReactNode;
}

export function Tag({ tone = "neutral", children }: TagProps) {
  const color = tone === "warning" ? "var(--warning)" : "var(--text-3)";
  return (
    <span
      className="mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontSize: "var(--step--1)",
        lineHeight: 1,
        padding: "2px var(--space-2)",
        borderRadius: "var(--radius-sm)",
        color,
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      {children}
    </span>
  );
}
