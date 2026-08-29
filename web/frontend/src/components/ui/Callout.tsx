import type { ReactNode } from "react";
import { Info, CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

export interface CalloutProps {
  tone: "info" | "success" | "warning" | "danger";
  title?: ReactNode;
  icon?: ReactNode;
  children: ReactNode;
}

const TONE: Record<CalloutProps["tone"], { color: string; icon: ReactNode }> = {
  info: { color: "var(--info)", icon: <Info size={18} /> },
  success: { color: "var(--success)", icon: <CheckCircle2 size={18} /> },
  warning: { color: "var(--warning)", icon: <AlertTriangle size={18} /> },
  danger: { color: "var(--danger)", icon: <XCircle size={18} /> },
};

export function Callout({ tone, title, icon, children }: CalloutProps) {
  const t = TONE[tone];
  return (
    <div
      role="note"
      style={{
        display: "flex",
        gap: "var(--space-3)",
        padding: "var(--space-4)",
        borderRadius: "var(--radius-md)",
        borderLeft: `3px solid ${t.color}`,
        background: `color-mix(in srgb, ${t.color} 10%, transparent)`,
        color: "var(--text-2)",
      }}
    >
      <span style={{ color: t.color, flexShrink: 0, display: "inline-flex", marginTop: "2px" }}>
        {icon ?? t.icon}
      </span>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
        {title && <strong style={{ color: "var(--text-1)" }}>{title}</strong>}
        <div>{children}</div>
      </div>
    </div>
  );
}
