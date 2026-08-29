import { Field, Select } from "../ui";
import type { UseCase } from "../../types/api";

export interface ObjectivePickerProps {
  useCase?: UseCase;
  value: string;
  onChange: (obj: string) => void;
}

export function ObjectivePicker({ useCase, value, onChange }: ObjectivePickerProps) {
  const objectives = useCase?.compatible_objectives ?? ["supervised"];
  const desc = useCase?.data_formats?.[value]?.user_facing_description;
  return (
    <Field label="Objective">
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {objectives.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </Select>
      {desc && <span style={{ fontSize: "13px", color: "var(--text-3)" }}>{desc}</span>}
    </Field>
  );
}
