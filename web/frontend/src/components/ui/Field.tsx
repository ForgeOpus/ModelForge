import type { ReactNode } from "react";

export interface FieldProps {
  label: string;
  hint?: string;
  error?: string | null;
  htmlFor?: string;
  children: ReactNode;
}

export function Field({ label, hint, error, htmlFor, children }: FieldProps) {
  return (
    <div className="stack-sm" style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
      <label
        htmlFor={htmlFor}
        style={{ fontSize: "13px", color: "var(--text-3)", fontWeight: 500 }}
      >
        {label}
      </label>
      {children}
      {error ? (
        <span style={{ fontSize: "13px", color: "var(--danger)" }}>{error}</span>
      ) : hint ? (
        <span style={{ fontSize: "13px", color: "var(--text-3)" }}>{hint}</span>
      ) : null}
    </div>
  );
}
