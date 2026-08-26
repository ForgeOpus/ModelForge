import { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import { motion } from "motion/react";

import { GPU_PRESETS, adjustedVram, selectTier } from "@/lib/vram";
import { TIERS, TIER_ORDER } from "@/lib/tiers";
import { formatParams } from "@/lib/utils";
import { VramMeter } from "@/components/VramMeter";
import { EASE } from "@/lib/motion";

/* The hero is the product's actual claim, made checkable.

   Every other tool in this space opens with a headline over a gradient. The
   one honest thing ModelForge can say up front is "here is what your card can
   train", so the hero answers that -- with the engine's own arithmetic, before
   anything is installed. It is the same budget meter the app uses, at its
   largest scale. */

const LORA_RANK = (params: number) =>
  params < 1e9 ? 8 : params < 7e9 ? 16 : params < 13e9 ? 32 : 64;

export function HeroMatcher() {
  const [index, setIndex] = useState(4); // RTX 3090, a common mid-range default
  const preset = GPU_PRESETS[index];

  const result = useMemo(() => {
    const cpuOnly = preset.name === "CPU only";
    const mps = preset.name.startsWith("Apple");
    const raw = preset.vramGb;
    const adjusted = adjustedVram(raw);
    const selection = selectTier(adjusted, { cpuOnly, mps, ramGb: 32 });
    // Show a model that comfortably fits rather than one exactly at the limit.
    const usable = selection.maxParams * 0.92;
    return { selection, usable, cpuOnly, mps };
  }, [preset]);

  // The engine always prefers quality, so on a big card it recommends a small
  // full finetune. That is genuinely what it does -- but someone asking "what
  // fits on my 4090" also wants the other end of the trade.
  const cheapest = useMemo(() => {
    const entries = TIER_ORDER.map((t) => ({ tier: t, ceiling: result.selection.byTier[t] }));
    const best = entries.reduce((a, b) => (b.ceiling > a.ceiling ? b : a));
    return best.tier !== result.selection.tier && best.ceiling > 0 ? best : null;
  }, [result]);

  const totalGb = preset.vramGb || 8;
  const meta = TIERS[result.selection.tier];

  return (
    <div className="rounded-lg border border-line bg-ground p-5 shadow-[var(--shadow-card)]">
      <label className="mb-4 block">
        <span className="eyebrow mb-1.5 block">Your graphics card</span>
        <div className="relative">
          <select
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
            className="w-full appearance-none rounded-sm border border-line bg-surface py-2.5 pl-3 pr-9 font-mono text-[13px] text-text"
          >
            {GPU_PRESETS.map((g, i) => (
              <option key={g.name} value={i}>
                {g.name}
                {g.vramGb > 0 ? ` · ${g.vramGb} GB` : ""}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-faint" />
        </div>
      </label>

      {result.cpuOnly ? (
        <p className="text-[13px] leading-relaxed text-dim">
          Without a GPU, ModelForge caps you at roughly{" "}
          <span className="font-mono text-text">500M parameters</span> and training takes days
          rather than minutes. It will still run, and it will tell you exactly what to expect.
        </p>
      ) : (
        <>
          <VramMeter
            numParams={result.usable}
            peft={result.selection.tier}
            totalGb={totalGb}
            measureKey={index}
          />

          <motion.div
            key={index}
            className="mt-4 border-t border-line pt-3.5"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.28, ease: EASE, delay: 0.5 }}
          >
            <p className="text-[13px] leading-relaxed text-text">
              ModelForge would pick{" "}
              <b className="font-mono" style={{ color: meta.cssVar }}>
                {meta.label}
              </b>{" "}
              here and train a model up to{" "}
              <b className="font-mono" style={{ color: meta.cssVar }}>
                {formatParams(result.selection.maxParams)}
              </b>{" "}
              parameters
              {result.selection.tier !== "full_finetune" && result.usable > 0
                ? ` with a rank-${LORA_RANK(result.usable)} adapter.`
                : "."}
            </p>
            {cheapest && (
              <p className="mt-1.5 text-[12px] leading-relaxed text-faint">
                Trading precision for size,{" "}
                <b className="font-mono" style={{ color: TIERS[cheapest.tier].cssVar }}>
                  {cheapest.tier}
                </b>{" "}
                fits up to {formatParams(cheapest.ceiling)} on the same card.
              </p>
            )}
            {preset.note && <p className="mt-1.5 text-[11.5px] text-faint">{preset.note}</p>}
          </motion.div>
        </>
      )}
    </div>
  );
}
