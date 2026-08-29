import type { ReactNode } from "react";

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
}

export function EmptyState({ icon, title, children }: EmptyStateProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        gap: "var(--space-3)",
        padding: "var(--space-10) var(--space-6)",
        color: "var(--text-3)",
      }}
    >
      {icon && <span style={{ display: "inline-flex", color: "var(--text-3)" }}>{icon}</span>}
      <strong style={{ color: "var(--text-1)", fontSize: "var(--step-1)" }}>{title}</strong>
      {children && <div style={{ maxWidth: "42ch" }}>{children}</div>}
    </div>
  );
}
