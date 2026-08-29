import { Field, Checkbox } from "../ui";
import type { Gpu } from "../../types/api";

export interface GpuSelectorProps {
  gpus: Gpu[];
  selected: number[];
  onChange: (indices: number[]) => void;
}

export function GpuSelector({ gpus, selected, onChange }: GpuSelectorProps) {
  if (gpus.length <= 1) return null;

  const toggle = (i: number, on: boolean) => {
    const next = on ? [...selected, i] : selected.filter((x) => x !== i);
    next.sort((a, b) => a - b);
    onChange(next);
  };

  return (
    <Field label="GPUs">
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
        {gpus.map((gpu, i) => {
          const mem =
            gpu.total_memory_gb === "N/A" ? "shared" : `${gpu.total_memory_gb} GB`;
          return (
            <Checkbox
              key={i}
              label={`${i}: ${gpu.name} (${mem})`}
              checked={selected.includes(i)}
              onChange={(e) => toggle(i, e.target.checked)}
            />
          );
        })}
      </div>
      {selected.length === 0 && (
        <span style={{ fontSize: "13px", color: "var(--text-3)" }}>
          No selection — all GPUs will be used.
        </span>
      )}
    </Field>
  );
}
