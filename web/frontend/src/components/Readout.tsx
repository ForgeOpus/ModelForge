import { useTween } from "@/lib/motion";
import { cn } from "@/lib/utils";

/* A number that sweeps to its new value.

   Used for every instrument figure -- the VRAM budget, loss, gradient norm,
   step counts. Tabular figures keep the digits from shifting sideways while
   the value moves. */
export function Readout({
  value,
  digits = 0,
  suffix,
  className,
  duration,
}: {
  value: number;
  digits?: number;
  suffix?: string;
  className?: string;
  duration?: number;
}) {
  const shown = useTween(value, duration);
  return (
    <span className={cn("tnum tabular-nums", className)}>
      {shown.toFixed(digits)}
      {suffix}
    </span>
  );
}
