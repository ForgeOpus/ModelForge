import type { ReactNode } from "react";

export interface RationaleRowProps {
  label: string;
  children: ReactNode;
  mono?: boolean;
}

export function RationaleRow({ label, children, mono }: RationaleRowProps) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "var(--space-2) var(--space-4)",
        alignItems: "baseline",
        padding: "var(--space-3) 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span
        style={{
          fontSize: "var(--step--1)",
          textTransform: "uppercase",
          letterSpacing: "var(--tracking-wide)",
          color: "var(--text-3)",
          flexShrink: 0,
        }}
      >
        {label}
      </span>
      <span
        className={mono ? "mono" : undefined}
        style={{ color: "var(--text-2)", lineHeight: "var(--leading-normal)" }}
      >
        {children}
      </span>
    </div>
  );
}
