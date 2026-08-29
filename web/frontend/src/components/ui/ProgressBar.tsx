import type { ReactNode } from "react";
import "./ProgressBar.css";

export interface ProgressBarProps {
  value: number;
  label?: ReactNode;
  indeterminate?: boolean;
}

export function ProgressBar({ value, label, indeterminate = false }: ProgressBarProps) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
      {label && (
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--step--1)", color: "var(--text-3)" }}>
          <span>{label}</span>
          {!indeterminate && <span className="mono">{Math.round(pct)}%</span>}
        </div>
      )}
      <div
        className="ui-progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={indeterminate ? undefined : 1}
        aria-valuenow={indeterminate ? undefined : Math.max(0, Math.min(1, value))}
      >
        <div
          className={indeterminate ? "ui-progress-fill ui-progress-fill--indeterminate" : "ui-progress-fill"}
          style={indeterminate ? undefined : { width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
