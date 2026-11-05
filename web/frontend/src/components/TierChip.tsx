import * as Tooltip from "@radix-ui/react-tooltip";
import { tier, bytesLabel } from "@/lib/tiers";
import { cn } from "@/lib/utils";

/* Colour means precision, everywhere. A tier chip reads the same in the plan
   column, in the history table, and beside a saved model. */
export function TierChip({
  peft,
  className,
  showLabel = false,
}: {
  peft: string | null | undefined;
  className?: string;
  showLabel?: boolean;
}) {
  const meta = tier(peft);
  if (!meta) {
    return <span className={cn("font-mono text-[11px] text-faint", className)}>—</span>;
  }
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-xs border px-1.5 py-0.5 font-mono text-[11px]",
            className,
          )}
          style={{ color: meta.cssVar, borderColor: meta.cssVar }}
        >
          {showLabel ? meta.label : meta.slug}
        </span>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          side="top"
          sideOffset={6}
          className="z-50 max-w-[240px] rounded-sm border border-line bg-surface px-2.5 py-2 text-[12px] leading-relaxed text-dim shadow-[var(--shadow-card)]"
        >
          <span className="text-text">{meta.label}</span> — {meta.blurb}
          <div className="mt-1 font-mono text-[11px] text-faint">
            {bytesLabel(meta.bytesPerParam)}
          </div>
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function StatusChip({ status, muted }: { status: string; muted?: boolean }) {
  const color =
    muted || status === "cancelled"
      ? "var(--faint)"
      : status === "completed"
        ? "var(--ok)"
        : status === "failed"
          ? "var(--danger)"
          : status === "running"
            ? "var(--brand)"
            : "var(--dim)";
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[11px]" style={{ color }}>
      <span
        className={cn("size-1.5 rounded-full", status === "running" && "animate-pulse")}
        style={{ background: color }}
      />
      {status}
    </span>
  );
}
