import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Label({ className, ...props }: ComponentProps<"label">) {
  return <label className={cn("block text-[12px] text-dim mb-1.5", className)} {...props} />;
}

export function Input({ className, ...props }: ComponentProps<"input">) {
  return (
    <input
      className={cn(
        "w-full rounded-sm border border-line bg-ground px-2.5 py-2 font-mono text-[12.5px] text-text",
        "placeholder:text-faint focus-visible:outline-2 focus-visible:outline-brand focus-visible:outline-offset-[-1px]",
        "disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

export function Textarea({ className, ...props }: ComponentProps<"textarea">) {
  return (
    <textarea
      className={cn(
        "w-full resize-none rounded-sm border border-line bg-ground px-2.5 py-2 text-[13px] text-text",
        "placeholder:text-faint focus-visible:outline-2 focus-visible:outline-brand focus-visible:outline-offset-[-1px]",
        className,
      )}
      {...props}
    />
  );
}

/** A labelled row for the Advanced panel: name, control, and an optional
    marker showing the value no longer matches what the engine proposed. */
export function Row({
  label,
  hint,
  changed,
  onReset,
  children,
}: {
  label: string;
  hint?: string;
  changed?: boolean;
  onReset?: () => void;
  children: ReactNode;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_140px] items-center gap-3 py-1.5">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[12px] text-text truncate">{label}</span>
          {changed && (
            <button
              type="button"
              onClick={onReset}
              title="Reset to the recommended value"
              className="text-[10px] uppercase tracking-wider text-warn hover:underline"
            >
              changed
            </button>
          )}
        </div>
        {hint && <p className="text-[11px] text-faint leading-snug mt-0.5">{hint}</p>}
      </div>
      {children}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex rounded-sm border border-line bg-ground p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-xs px-3 py-1 font-mono text-[11.5px] transition-colors",
            value === o.value ? "bg-raised text-text" : "text-dim hover:text-text",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full border transition-colors",
        checked ? "border-brand bg-brand" : "border-line bg-track",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 size-3.5 rounded-full transition-all",
          checked ? "left-4.5 bg-brand-ink" : "left-0.5 bg-line-strong",
        )}
      />
    </button>
  );
}
