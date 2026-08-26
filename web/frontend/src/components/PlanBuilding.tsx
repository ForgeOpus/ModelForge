import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { EASE } from "@/lib/motion";
import { Panel } from "./ui/misc";
import { TIER_ORDER, TIERS } from "@/lib/tiers";

/* What the engine is actually doing while you wait.

   Building a plan takes a couple of seconds -- it reads the hardware, works
   out a ceiling for each precision tier, then queries the Hub. A spinner
   would hide all of that. Naming each step instead turns dead time into the
   clearest explanation of how the tool works that a first-time user will get,
   and it sets up the reading that follows. */

const STEPS = [
  "Reading your hardware",
  "Working out what each precision tier can hold",
  "Asking the Hub which models fit",
] as const;

export function PlanBuilding() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    // Paced to the real work: hardware detection is quick, the Hub query is not.
    const timers = [
      setTimeout(() => setStep(1), 550),
      setTimeout(() => setStep(2), 1300),
    ];
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <Panel className="bg-surface p-4">
      <div className="flex flex-col gap-2.5">
        {STEPS.map((label, i) => (
          <div key={label} className="flex items-center gap-2.5">
            <span className="relative flex size-3 shrink-0 items-center justify-center">
              {i < step ? (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.2, ease: EASE }}
                  className="size-1.5 rounded-full bg-ok"
                />
              ) : i === step ? (
                <motion.span
                  className="size-1.5 rounded-full bg-brand"
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                />
              ) : (
                <span className="size-1.5 rounded-full bg-line-strong" />
              )}
            </span>
            <span
              className={
                i <= step ? "text-[12.5px] text-text" : "text-[12.5px] text-faint"
              }
            >
              {label}
            </span>
          </div>
        ))}
      </div>

      {/* The tier ladder, drawn empty. It is the scale the reading lands on. */}
      <div className="mt-4 flex flex-col gap-1.5 border-t border-line pt-3.5">
        {TIER_ORDER.map((t, i) => (
          <div key={t} className="grid grid-cols-[92px_minmax(0,1fr)] items-center gap-2.5">
            <span className="font-mono text-[10px] text-faint">{t}</span>
            <span className="h-[3px] overflow-hidden rounded-full bg-track">
              <motion.span
                className="block h-full origin-left rounded-full"
                style={{ background: TIERS[t].cssVar, opacity: 0.35 }}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: step >= 1 ? 1 : 0 }}
                transition={{ duration: 0.5, ease: EASE, delay: i * 0.07 }}
                // scaleX animates from the left so it reads as filling, not growing.
              />
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
