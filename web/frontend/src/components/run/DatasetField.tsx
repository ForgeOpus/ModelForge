import { Field, Input, Checkbox, Button, Callout, Tag } from "../ui";
import type { DatasetValidationResult } from "../../types/api";

export interface DatasetFieldProps {
  path: string;
  remote: boolean;
  onPathChange: (path: string) => void;
  onRemoteChange: (remote: boolean) => void;
  onValidate: () => void;
  validation: DatasetValidationResult | null;
  validating: boolean;
  validationError?: string | null;
}

export function DatasetField({
  path,
  remote,
  onPathChange,
  onRemoteChange,
  onValidate,
  validation,
  validating,
  validationError,
}: DatasetFieldProps) {
  return (
    <Field label="Dataset">
      <Input
        value={path}
        placeholder="./data/train.jsonl or an HF dataset id"
        onChange={(e) => onPathChange(e.target.value)}
      />
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "var(--space-3)" }}>
        <Checkbox
          label="HuggingFace dataset"
          checked={remote}
          onChange={(e) => onRemoteChange(e.target.checked)}
        />
        <Button variant="ghost" loading={validating} onClick={onValidate}>
          Validate
        </Button>
      </div>

      {validationError ? (
        <Callout tone="danger">{validationError}</Callout>
      ) : validation ? (
        <Callout tone={validation.valid ? "success" : "danger"}>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
            {validation.columns.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-1)" }}>
                {validation.columns.map((c) => (
                  <Tag key={c}>{c}</Tag>
                ))}
              </div>
            )}
            {validation.format_description && <span>{validation.format_description}</span>}
            {validation.error && (
              <span style={{ whiteSpace: "pre-wrap" }}>{validation.error}</span>
            )}
          </div>
        </Callout>
      ) : null}
    </Field>
  );
}
