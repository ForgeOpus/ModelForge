import type { PeftApproach } from "@/types";

/* The PEFT tiers form a precision ladder. These bytes-per-parameter figures
   are the engine's own, from utilities/recommendation_engine.py — the same
   numbers that decide which models fit. Colour runs cold (expensive, high
   precision) to warm (cheap, low precision) so a run's cost is legible at a
   glance anywhere a tier is shown. */
export interface TierMeta {
  slug: PeftApproach;
  label: string;
  /** Bytes per parameter for model weights at this tier. */
  bytesPerParam: number;
  /** Weight bytes/param used for the disk-footprint warning; full finetune
      stores fp32 weights but trains with optimizer state on top. */
  weightBytesPerParam: number;
  cssVar: string;
  blurb: string;
}

/** "1 byte", not "1 bytes". */
export function bytesLabel(n: number): string {
  return `${n} ${n === 1 ? "byte" : "bytes"} per parameter`;
}

export const TIERS: Record<PeftApproach, TierMeta> = {
  full_finetune: {
    slug: "full_finetune",
    label: "full finetune",
    bytesPerParam: 18,
    weightBytesPerParam: 4,
    cssVar: "var(--tier-full)",
    blurb: "Every weight is updated. Highest quality, highest cost.",
  },
  lora_bf16: {
    slug: "lora_bf16",
    label: "LoRA bf16",
    bytesPerParam: 2,
    weightBytesPerParam: 2,
    cssVar: "var(--tier-lora)",
    blurb: "Base weights stay frozen at bf16; a small adapter is trained.",
  },
  qlora_8bit: {
    slug: "qlora_8bit",
    label: "QLoRA 8-bit",
    bytesPerParam: 1,
    weightBytesPerParam: 1,
    cssVar: "var(--tier-8bit)",
    blurb: "Base weights quantised to 8-bit, adapter trained on top.",
  },
  qlora_4bit: {
    slug: "qlora_4bit",
    label: "QLoRA 4-bit",
    bytesPerParam: 0.5,
    weightBytesPerParam: 0.5,
    cssVar: "var(--tier-4bit)",
    blurb: "Base weights quantised to 4-bit. Cheapest way to fit a large model.",
  },
};

export const TIER_ORDER: PeftApproach[] = [
  "full_finetune",
  "lora_bf16",
  "qlora_8bit",
  "qlora_4bit",
];

export function tier(slug: string | null | undefined): TierMeta | null {
  if (!slug) return null;
  return TIERS[slug as PeftApproach] ?? null;
}

export function tierColor(slug: string | null | undefined): string {
  return tier(slug)?.cssVar ?? "var(--dim)";
}
