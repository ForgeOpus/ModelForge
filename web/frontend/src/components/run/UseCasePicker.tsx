import { Field, Select } from "../ui";
import type { UseCase } from "../../types/api";

export interface UseCasePickerProps {
  useCases: UseCase[];
  value: string;
  onChange: (slug: string) => void;
}

export function UseCasePicker({ useCases, value, onChange }: UseCasePickerProps) {
  const selected = useCases.find((u) => u.slug === value);
  return (
    <Field label="Use case">
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {useCases.map((u) => (
          <option key={u.slug} value={u.slug}>
            {u.name}
          </option>
        ))}
      </Select>
      {selected?.description && (
        <span style={{ fontSize: "13px", color: "var(--text-3)" }}>{selected.description}</span>
      )}
    </Field>
  );
}
