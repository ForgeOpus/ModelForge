import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Input, Row, Toggle } from "@/components/ui/field";
import { Eyebrow } from "@/components/ui/misc";
import type { Recommendation } from "@/types";

export type Overrides = Record<string, unknown>;

/* Full parity with the old settings page.

   The API takes the whole recommendation object back verbatim, so every field
   below is already reachable -- it just needed a control. Values start at
   whatever the engine chose; anything you touch is marked so it's obvious what
   is yours and what is the plan's. */

type FieldKind = "number" | "text" | "bool" | "choice";

interface FieldSpec {
  key: string;
  label: string;
  kind: FieldKind;
  hint?: string;
  step?: number;
  min?: number;
  choices?: string[];
}

const GROUPS: { heading: string; blurb: string; fields: FieldSpec[] }[] = [
  {
    heading: "Schedule",
    blurb: "How long training runs and how fast it moves.",
    fields: [
      { key: "num_train_epochs", label: "num_train_epochs", kind: "number", step: 0.5, min: 0, hint: "Passes over the dataset." },
      { key: "learning_rate", label: "learning_rate", kind: "number", step: 0.000001, min: 0 },
      { key: "lr_scheduler_type", label: "lr_scheduler_type", kind: "choice", choices: ["linear", "cosine", "constant"] },
      { key: "warmup_ratio", label: "warmup_ratio", kind: "number", step: 0.01, min: 0 },
      { key: "weight_decay", label: "weight_decay", kind: "number", step: 0.001, min: 0 },
      { key: "max_grad_norm", label: "max_grad_norm", kind: "number", step: 0.1, min: 0 },
      { key: "optim", label: "optim", kind: "choice", choices: ["adamw_torch", "adamw_torch_fused", "paged_adamw_32bit", "adafactor"] },
      { key: "seed", label: "seed", kind: "number", step: 1, min: 0 },
    ],
  },
  {
    heading: "Batching",
    blurb: "Trades memory against throughput. The engine sizes these from your VRAM.",
    fields: [
      { key: "per_device_train_batch_size", label: "per_device_train_batch_size", kind: "number", step: 1, min: 1 },
      { key: "gradient_accumulation_steps", label: "gradient_accumulation_steps", kind: "number", step: 1, min: 1 },
      { key: "max_length", label: "max_length", kind: "number", step: 128, min: 32, hint: "Sequence length. Longer costs more activation memory." },
      { key: "gradient_checkpointing", label: "gradient_checkpointing", kind: "bool", hint: "Recompute activations to save memory." },
      { key: "activation_offloading", label: "activation_offloading", kind: "bool" },
      { key: "packing", label: "packing", kind: "bool", hint: "Pack short examples together." },
    ],
  },
  {
    heading: "Precision",
    blurb: "Changing these moves the memory budget. Rebuild the plan afterwards to re-check the fit.",
    fields: [
      { key: "bf16", label: "bf16", kind: "bool" },
      { key: "fp16", label: "fp16", kind: "bool" },
      { key: "use_liger_kernel", label: "use_liger_kernel", kind: "bool" },
    ],
  },
  {
    heading: "Checkpoints",
    blurb: "How often progress is written to disk and reported.",
    fields: [
      { key: "save_steps", label: "save_steps", kind: "number", step: 50, min: 1 },
      { key: "logging_steps", label: "logging_steps", kind: "number", step: 1, min: 1, hint: "Also sets how often the loss chart updates." },
    ],
  },
];

const LORA_FIELDS: FieldSpec[] = [
  { key: "peft.r", label: "r", kind: "number", step: 8, min: 1, hint: "Adapter rank." },
  { key: "peft.lora_alpha", label: "lora_alpha", kind: "number", step: 8, min: 1 },
];

export function AdvancedPanel({
  recommendation,
  overrides,
  onChange,
}: {
  recommendation: Recommendation;
  overrides: Overrides;
  onChange: (next: Overrides) => void;
}) {
  const [open, setOpen] = useState(false);

  const baseValue = (key: string): unknown => {
    if (key.startsWith("peft.")) {
      return (recommendation.peft_config as Record<string, unknown> | null)?.[key.slice(5)];
    }
    return (recommendation.strategy_config as Record<string, unknown>)?.[key];
  };

  const current = (key: string): unknown =>
    key in overrides ? overrides[key] : baseValue(key);

  const set = (key: string, value: unknown) => {
    const next = { ...overrides };
    if (Object.is(value, baseValue(key))) delete next[key];
    else next[key] = value;
    onChange(next);
  };

  const reset = (key: string) => {
    const next = { ...overrides };
    delete next[key];
    onChange(next);
  };

  const groups = recommendation.peft_config
    ? [...GROUPS, { heading: "LoRA adapter", blurb: "Capacity of the trained adapter.", fields: LORA_FIELDS }]
    : GROUPS;

  return (
    <div className="mt-4 rounded-sm border border-line">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-[12.5px] text-dim hover:text-text"
      >
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
        Advanced settings
        <span className="ml-auto font-mono text-[11px] text-faint">
          {Object.keys(overrides).length > 0
            ? `${Object.keys(overrides).length} changed`
            : "using the plan"}
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-5 border-t border-line px-3 py-4">
          {groups.map((group) => (
            <div key={group.heading}>
              <Eyebrow className="mb-1">{group.heading}</Eyebrow>
              <p className="mb-2 text-[11.5px] leading-snug text-faint">{group.blurb}</p>
              <div className="divide-y divide-line">
                {group.fields
                  .filter((f) => baseValue(f.key) !== undefined)
                  .map((f) => (
                    <Row
                      key={f.key}
                      label={f.label}
                      hint={f.hint}
                      changed={f.key in overrides}
                      onReset={() => reset(f.key)}
                    >
                      <Control spec={f} value={current(f.key)} onChange={(v) => set(f.key, v)} />
                    </Row>
                  ))}
              </div>
            </div>
          ))}

          <p className="text-[11.5px] leading-relaxed text-faint">
            Quantisation is set by the precision tier and is not editable here — change it by
            picking a different model or freeing VRAM, then rebuilding the plan.
          </p>
        </div>
      )}
    </div>
  );
}

function Control({
  spec,
  value,
  onChange,
}: {
  spec: FieldSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (spec.kind === "bool") {
    return (
      <div className="flex justify-end">
        <Toggle checked={Boolean(value)} onChange={onChange} label={spec.label} />
      </div>
    );
  }
  if (spec.kind === "choice") {
    return (
      <select
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
        aria-label={spec.label}
        className="w-full rounded-sm border border-line bg-ground px-2 py-1.5 font-mono text-[11.5px] text-text"
      >
        {spec.choices?.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    );
  }
  return (
    <Input
      type={spec.kind === "number" ? "number" : "text"}
      step={spec.step}
      min={spec.min}
      value={value === undefined || value === null ? "" : String(value)}
      aria-label={spec.label}
      className="py-1.5 text-right"
      onChange={(e) => {
        if (spec.kind === "number") {
          const n = Number(e.target.value);
          onChange(e.target.value === "" ? null : Number.isNaN(n) ? e.target.value : n);
        } else {
          onChange(e.target.value);
        }
      }}
    />
  );
}
