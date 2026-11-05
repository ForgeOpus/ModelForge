import * as DialogPrimitive from "@radix-ui/react-dialog";
import { AlertTriangle, X } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Panel({ className, ...props }: ComponentProps<"section">) {
  return (
    <section
      className={cn("rounded-md border border-line bg-ground", className)}
      {...props}
    />
  );
}

export function Eyebrow({ className, ...props }: ComponentProps<"h3">) {
  return <h3 className={cn("eyebrow", className)} {...props} />;
}

export function Chip({
  className,
  color,
  ...props
}: ComponentProps<"span"> & { color?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-xs border px-1.5 py-0.5 font-mono text-[11px]",
        className,
      )}
      style={color ? { color, borderColor: color } : undefined}
      {...props}
    />
  );
}

export function Alert({
  tone = "warn",
  title,
  children,
}: {
  tone?: "warn" | "danger" | "ok";
  title?: string;
  children: ReactNode;
}) {
  const map = {
    warn: "var(--warn)",
    danger: "var(--danger)",
    ok: "var(--ok)",
  } as const;
  const c = map[tone];
  return (
    <div
      className="flex gap-2.5 rounded-sm border p-2.5 text-[12.5px] leading-relaxed"
      style={{
        borderColor: `color-mix(in srgb, ${c} 38%, transparent)`,
        background: `color-mix(in srgb, ${c} 10%, var(--surface))`,
      }}
    >
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" style={{ color: c }} />
      <div className="min-w-0">
        {title && <p className="font-medium text-text mb-0.5">{title}</p>}
        <div className="text-dim">{children}</div>
      </div>
    </div>
  );
}

/** An empty screen is an invitation to act, so it always names the next step. */
export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      {icon && <div className="text-faint [&_svg]:size-7">{icon}</div>}
      <h3 className="font-display text-[16px] font-semibold">{title}</h3>
      {children && <p className="max-w-sm text-[13px] leading-relaxed text-dim">{children}</p>}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-sm bg-raised", className)} />;
}

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;

export function DialogContent({
  title,
  description,
  children,
  className,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/55 backdrop-blur-[2px] data-[state=open]:animate-in data-[state=open]:fade-in" />
      <DialogPrimitive.Content
        className={cn(
          "fixed left-1/2 top-1/2 z-50 w-[min(92vw,440px)] -translate-x-1/2 -translate-y-1/2",
          "rounded-lg border border-line bg-surface p-5 shadow-[var(--shadow-card)]",
          className,
        )}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <DialogPrimitive.Title className="font-display text-[15px] font-semibold">
              {title}
            </DialogPrimitive.Title>
            {description && (
              <DialogPrimitive.Description className="mt-1 text-[12.5px] leading-relaxed text-dim">
                {description}
              </DialogPrimitive.Description>
            )}
          </div>
          <DialogPrimitive.Close className="rounded-xs p-1 text-faint hover:text-text">
            <X className="size-4" />
            <span className="sr-only">Close</span>
          </DialogPrimitive.Close>
        </div>
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export const DialogClose = DialogPrimitive.Close;
