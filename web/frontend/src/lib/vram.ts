import type { Gpu, HardwareSpecs, PeftApproach } from "@/types";
import { TIERS } from "./tiers";

/* A faithful port of utilities/recommendation_engine.py.

   The server is always the authority — when a Recommendation carries a
   vram_estimate we render that. This exists so the landing page can answer
   "what fits on my GPU?" before anything is installed, and so the budget
   meter can respond instantly while you flip between candidate models
   rather than waiting on a round trip.

   Keep these constants in step with the engine. */
export const ACTIVATION_BUDGET_GB = 2.0;
export const ADAPTER_COST_GB = 0.1;

/** The detector emits the string "N/A" for Apple Silicon, which reports no
    queryable VRAM. Anything non-numeric contributes nothing to the sum. */
export function gpuVram(gpu: Gpu): number | null {
  return typeof gpu.total_memory_gb === "number" ? gpu.total_memory_gb : null;
}

export function rawVram(hardware: HardwareSpecs | null): number {
  if (!hardware) return 0;
  return hardware.gpus.reduce((sum, g) => sum + (gpuVram(g) ?? 0), 0);
}

export function isCpuOnly(hardware: HardwareSpecs | null): boolean {
  return !hardware || hardware.gpus.length === 0;
}

/** Apple Silicon: one GPU entry with a non-numeric VRAM figure. */
export function isMps(hardware: HardwareSpecs | null): boolean {
  if (!hardware) return false;
  return hardware.gpus.some((g) => gpuVram(g) === null);
}

/** Strategy overhead. SFT is (0, 1.0); DPO holds a reference model and
    doubles the batch, so (1, 2.0). */
export function adjustedVram(
  rawVramGb: number,
  extraModelCopies = 0,
  batchMultiplier = 1.0,
): number {
  const actBytes = ACTIVATION_BUDGET_GB * 1e9;
  const adapterBytes = ADAPTER_COST_GB * 1e9;
  const rawBytes = rawVramGb * 1e9;

  // Bootstrap at 4-bit to break the circular model-size/tier dependency.
  let pBootstrap = (rawBytes - actBytes - adapterBytes) / 0.5;
  if (pBootstrap < 0) pBootstrap = 0;
  const estimatedWeightGb = (0.5 * pBootstrap) / 1e9;

  return (
    rawVramGb -
    extraModelCopies * estimatedWeightGb -
    (batchMultiplier - 1.0) * ACTIVATION_BUDGET_GB
  );
}

export interface TierSelection {
  tier: PeftApproach;
  maxParams: number;
  byTier: Record<PeftApproach, number>;
}

/** Mirrors _select_peft_tier: prefer quality, taking the first tier whose
    parameter ceiling is above zero. */
export function selectTier(
  adjustedVramGb: number,
  opts: { cpuOnly?: boolean; mps?: boolean; ramGb?: number } = {},
): TierSelection {
  const { cpuOnly = false, mps = false, ramGb = 0 } = opts;

  if (cpuOnly) {
    const pMaxCpu = Math.min(500_000_000, (ramGb * 1e9) / 3);
    return {
      tier: "qlora_4bit",
      maxParams: pMaxCpu,
      byTier: { full_finetune: 0, lora_bf16: 0, qlora_8bit: 0, qlora_4bit: pMaxCpu },
    };
  }

  const act = ACTIVATION_BUDGET_GB;
  const adapter = ADAPTER_COST_GB;
  const byTier: Record<PeftApproach, number> = {
    full_finetune: Math.max(0, ((adjustedVramGb - act) / 18) * 1e9),
    lora_bf16: Math.max(0, ((adjustedVramGb - adapter - act) / 2) * 1e9),
    qlora_8bit: Math.max(0, ((adjustedVramGb - adapter - act) / 1) * 1e9),
    qlora_4bit: Math.max(0, ((adjustedVramGb - adapter - act) / 0.5) * 1e9),
  };

  if (mps) {
    // bitsandbytes has no MPS backend, so quantised tiers are unavailable.
    byTier.qlora_8bit = 0;
    byTier.qlora_4bit = 0;
    if (byTier.full_finetune > 0) {
      return { tier: "full_finetune", maxParams: byTier.full_finetune, byTier };
    }
    return { tier: "lora_bf16", maxParams: byTier.lora_bf16, byTier };
  }

  for (const t of ["full_finetune", "lora_bf16", "qlora_8bit", "qlora_4bit"] as PeftApproach[]) {
    if (byTier[t] > 0) return { tier: t, maxParams: byTier[t], byTier };
  }
  // Nothing fits; the engine reports 0 so the caller can warn.
  return { tier: "qlora_4bit", maxParams: 0, byTier };
}

export interface Segment {
  key: "weights" | "adapter" | "activations";
  label: string;
  gb: number;
  color: string;
}

export interface Footprint {
  segments: Segment[];
  /** Everything the run is expected to occupy. */
  usedGb: number;
  totalGb: number;
  freeGb: number;
  /** True when the projection exceeds the card. */
  over: boolean;
}

/** What a given model costs at a given tier, broken into the parts that
    consume it. This is the shape the budget meter renders at every scale. */
export function footprint(
  numParams: number,
  peft: PeftApproach,
  totalGb: number,
): Footprint {
  const meta = TIERS[peft];
  const weights = (numParams * meta.bytesPerParam) / 1e9;
  // A full finetune trains the weights themselves — there is no adapter.
  const adapter = peft === "full_finetune" ? 0 : ADAPTER_COST_GB;

  const segments: Segment[] = [
    {
      key: "weights",
      // At full precision this figure covers gradients and optimizer state too,
      // which is most of it -- calling it "weights" would understate the cost.
      label: peft === "full_finetune" ? "weights + optimizer" : "weights",
      gb: weights,
      color: meta.cssVar,
    },
    { key: "adapter", label: "adapter", gb: adapter, color: meta.cssVar },
    { key: "activations", label: "activations", gb: ACTIVATION_BUDGET_GB, color: "var(--activation)" },
  ].filter((s) => s.gb > 0) as Segment[];

  const usedGb = weights + adapter + ACTIVATION_BUDGET_GB;
  return {
    segments,
    usedGb,
    totalGb,
    freeGb: Math.max(0, totalGb - usedGb),
    over: usedGb > totalGb,
  };
}

/** Reference cards for the landing page's matcher, so a visitor can see the
    answer for their hardware before installing anything. */
export interface GpuPreset {
  name: string;
  vramGb: number;
  note?: string;
}

export const GPU_PRESETS: GpuPreset[] = [
  { name: "RTX 3060", vramGb: 12 },
  { name: "RTX 4070 Ti", vramGb: 12 },
  { name: "RTX 4080", vramGb: 16 },
  { name: "RTX 5060 Ti", vramGb: 16 },
  { name: "RTX 3090", vramGb: 24 },
  { name: "RTX 4090", vramGb: 24 },
  { name: "A100", vramGb: 40 },
  { name: "H100", vramGb: 80 },
  { name: "2 × RTX 4090", vramGb: 48, note: "VRAM is summed; sharding is not configured for you." },
  { name: "Apple M-series", vramGb: 0, note: "Unified memory. Quantisation is unavailable on MPS." },
  { name: "CPU only", vramGb: 0, note: "Training will be extremely slow." },
];
