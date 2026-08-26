import { useEffect, useRef } from "react";
import anime from "animejs";
import type { PeftApproach } from "@/types";
import { footprint } from "@/lib/vram";
import { TIERS } from "@/lib/tiers";
import { prefersReducedMotion } from "@/lib/motion";
import { Readout } from "./Readout";
import { cn } from "@/lib/utils";

/* The signature element, at four scales: ambient in the rail, inline in a
   model row, full in the plan column, and large on the landing page.

   ModelForge's whole premise is that one number -- available VRAM -- decides
   everything else, so the budget is drawn the same way wherever it appears.

   The full-size version also carries the app's one orchestrated moment. When a
   plan resolves, the meter behaves like an instrument taking a reading: the
   scale is laid down, a sweep runs the length of it, and only then do the
   segments settle into place. That sequence is the product's thesis made
   visible -- the machine measuring your hardware before it commits to
   anything. It is the only place in the app that animates for longer than a
   fifth of a second. */

export interface MeterProps {
  numParams: number;
  peft: PeftApproach;
  totalGb: number;
  className?: string;
}

/** Rail scale: a hairline bar, no labels. */
export function VramMicro({
  usedGb, totalGb, peft, className,
}: {
  usedGb: number; totalGb: number; peft?: PeftApproach | null; className?: string;
}) {
  const pct = totalGb > 0 ? Math.min(100, (usedGb / totalGb) * 100) : 0;
  const over = usedGb > totalGb;
  return (
    <div className={cn("h-[3px] overflow-hidden rounded-full bg-track", className)}>
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{
          width: `${pct}%`,
          background: over ? "var(--danger)" : peft ? TIERS[peft].cssVar : "var(--brand)",
        }}
      />
    </div>
  );
}

/** Model-row scale: proportion only, tinted by the tier it would run at. */
export function VramMini({ numParams, peft, totalGb, className }: MeterProps) {
  const f = footprint(numParams, peft, totalGb);
  const pct = totalGb > 0 ? Math.min(100, (f.usedGb / totalGb) * 100) : 0;
  return (
    <div className={cn("h-[3px] overflow-hidden rounded-full bg-track", className)}>
      <div
        className="h-full rounded-full transition-[width] duration-[450ms] ease-[cubic-bezier(.22,.8,.3,1)]"
        style={{ width: `${pct}%`, background: f.over ? "var(--danger)" : TIERS[peft].cssVar }}
      />
    </div>
  );
}

/** Plan-column and hero scale: segmented, ticked, labelled, and measured. */
export function VramMeter({
  numParams,
  peft,
  totalGb,
  size = "md",
  /** Bump to re-run the measurement sequence (a new plan, a new GPU). */
  measureKey,
  className,
}: MeterProps & { size?: "md" | "lg"; measureKey?: string | number }) {
  const f = footprint(numParams, peft, totalGb);
  const meta = TIERS[peft];
  const ticks = Math.max(1, Math.min(48, Math.round(totalGb)));
  const rootRef = useRef<HTMLDivElement>(null);
  const firstRun = useRef(true);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const ticksEls = root.querySelectorAll<HTMLElement>("[data-tick]");
    const segs = root.querySelectorAll<HTMLElement>("[data-seg]");
    const sweep = root.querySelector<HTMLElement>("[data-sweep]");

    if (prefersReducedMotion()) {
      ticksEls.forEach((el) => (el.style.opacity = "1"));
      segs.forEach((el) => (el.style.transform = "scaleX(1)"));
      if (sweep) sweep.style.opacity = "0";
      return;
    }

    // Subsequent readings just re-settle the segments; only the first reading
    // on a given plan gets the full instrument sweep.
    const full = firstRun.current || measureKey !== undefined;
    firstRun.current = false;

    const tl = anime.timeline({ easing: "cubicBezier(.22,.8,.3,1)" });

    if (full) {
      tl.add({
        targets: ticksEls,
        opacity: [0, 1],
        duration: 260,
        delay: anime.stagger(9),
      });
      if (sweep) {
        tl.add({
          targets: sweep,
          left: ["0%", "100%"],
          opacity: [0, 1, 1, 0],
          duration: 520,
          easing: "cubicBezier(.4,0,.2,1)",
        }, "-=120");
      }
    }

    tl.add({
      targets: segs,
      scaleX: [0, 1],
      duration: 560,
      delay: anime.stagger(80),
    }, full ? "-=180" : 0);

    return () => tl.pause();
  }, [numParams, peft, totalGb, measureKey]);

  return (
    <div className={className} ref={rootRef}>
      <div className="mb-3 flex items-baseline gap-2">
        <Readout
          value={f.usedGb}
          digits={1}
          className={cn(
            "font-display font-bold leading-none",
            size === "lg" ? "text-[42px]" : "text-[30px]",
          )}
        />
        <span className="font-mono text-[12px] text-faint">/ {totalGb.toFixed(1)} GB</span>
        <span
          className="ml-auto rounded-xs border px-1.5 py-0.5 font-mono text-[10.5px] transition-colors duration-300"
          style={{ color: meta.cssVar, borderColor: meta.cssVar }}
        >
          {peft}
        </span>
      </div>

      <div
        className={cn(
          "relative flex overflow-hidden rounded-xs bg-track",
          size === "lg" ? "h-3" : "h-2.5",
        )}
        role="img"
        aria-label={`${f.usedGb.toFixed(1)} of ${totalGb.toFixed(1)} gigabytes used at ${peft}`}
      >
        {f.segments.map((s) => (
          <div
            key={s.key}
            data-seg
            className="h-full origin-left"
            style={{
              width: `${Math.min(100, (s.gb / totalGb) * 100)}%`,
              background: s.color,
              opacity: s.key === "adapter" ? 0.5 : 1,
              marginRight: "1px",
            }}
          />
        ))}
        {/* The measuring sweep: one pass down the scale, then gone. */}
        <span
          data-sweep
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 w-8 opacity-0"
          style={{
            background: `linear-gradient(90deg, transparent, ${meta.cssVar}, transparent)`,
            filter: "blur(1px)",
          }}
        />
      </div>

      <div className="mt-1.5 flex" aria-hidden="true">
        {Array.from({ length: ticks }).map((_, i) => (
          <span
            key={i}
            data-tick
            className="h-[3px] flex-1"
            style={{ borderLeft: i === 0 ? "none" : "1px solid var(--line)" }}
          />
        ))}
      </div>

      <div className="mt-2.5 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10.5px] text-faint">
        {f.segments.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <b
              className="block size-[7px] rounded-[1px]"
              style={{ background: s.color, opacity: s.key === "adapter" ? 0.5 : 1 }}
            />
            {s.label} <Readout value={s.gb} digits={2} />
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <b className="block size-[7px] rounded-[1px] bg-track" />
          free <Readout value={f.freeGb} digits={2} />
        </span>
      </div>
    </div>
  );
}
