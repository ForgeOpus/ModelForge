import type { ReactNode } from "react";

export interface BadgeProps {
  color?: string;
  children: ReactNode;
}

export function Badge({ color = "var(--text-3)", children }: BadgeProps) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--space-1)",
        fontSize: "var(--step--1)",
        fontWeight: 600,
        lineHeight: 1,
        padding: "var(--space-1) var(--space-2)",
        borderRadius: "var(--radius-pill)",
        color: color,
        background: `color-mix(in srgb, ${color} 14%, transparent)`,
      }}
    >
      {children}
    </span>
  );
}
