/*
 * RunForm renders ONLY the form controls. The page owns the two-column layout
 * and renders <RationaleReport> itself on the right.
 *
 * State-lifting contract: RunForm calls `onStateChange` whenever the recommendation
 * or the selected model id changes, passing:
 *   { recommendation, selectedModelId, setSelectedModelId }
 * The page feeds `recommendation` + `selectedModelId` into <RationaleReport> and
 * wires its `onSelectModel` to `setSelectedModelId`. Picking a candidate on the
 * right updates `modelId` inside the form through that same setter.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Field, Input, Checkbox, Callout } from "../ui";
import { useSystem, useHardware, useRecommendation, useDatasetValidation } from "../../hooks";
import type { Recommendation, TrainingObjective } from "../../types/api";
import { UseCasePicker } from "./UseCasePicker";
import { ObjectivePicker } from "./ObjectivePicker";
import { GpuSelector } from "./GpuSelector";
import { DatasetField } from "./DatasetField";

export interface FormPayload {
  use_case_slug: string;
  objective: TrainingObjective;
  model_id: string;
  dataset_path: string;
  dataset_remote: boolean;
  output_dir: string;
  gpu_indices: number[];
  tensorboard: boolean;
  recommendation: Recommendation;
}

export interface RunFormProps {
  onSubmit: (payload: FormPayload) => Promise<void>;
  submitting: boolean;
  submitError?: string | null;
  onStateChange?: (s: {
    recommendation: Recommendation | null;
    selectedModelId: string;
    setSelectedModelId: (id: string) => void;
  }) => void;
}

export function RunForm({ onSubmit, submitting, submitError, onStateChange }: RunFormProps) {
  const system = useSystem();
  const hardware = useHardware();
  const rec = useRecommendation();
  const datasetValidation = useDatasetValidation();

  const useCases = useMemo(() => system.data?.use_cases ?? [], [system.data]);
  const gpus = hardware.data?.gpus ?? [];

  const [useCaseSlug, setUseCaseSlug] = useState("instruction_following");
  const [objective, setObjective] = useState("supervised");
  const [gpuIndices, setGpuIndices] = useState<number[]>([]);
  const [modelId, setModelId] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [datasetRemote, setDatasetRemote] = useState(false);
  const [outputDir, setOutputDir] = useState("./modelforge_output");
  const [tensorboard, setTensorboard] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState("");

  const [initialisedSlug, setInitialisedSlug] = useState(false);
  useEffect(() => {
    if (!initialisedSlug && useCases.length > 0) {
      setUseCaseSlug(useCases[0].slug);
      setInitialisedSlug(true);
    }
  }, [initialisedSlug, useCases]);

  const activeUseCase = useCases.find((u) => u.slug === useCaseSlug);

  const setSelectedAndModel = useCallback((id: string) => {
    setSelectedModelId(id);
    setModelId(id);
  }, []);

  // Keep the callback in a ref so the effect below never depends on its identity
  // (the parent typically passes an inline arrow). Firing it only when the real
  // values change avoids a render loop that would starve route transitions.
  const onStateChangeRef = useRef(onStateChange);
  onStateChangeRef.current = onStateChange;

  useEffect(() => {
    onStateChangeRef.current?.({
      recommendation: rec.recommendation,
      selectedModelId,
      setSelectedModelId: setSelectedAndModel,
    });
  }, [rec.recommendation, selectedModelId, setSelectedAndModel]);

  const handleGenerate = async () => {
    const result = await rec.generate({
      use_case_slug: useCaseSlug,
      objective: objective as TrainingObjective,
      gpu_indices: gpuIndices,
      n: 5,
    });
    const first = result?.recommended_models[0]?.model_id ?? "";
    if (first) {
      setModelId(first);
      setSelectedModelId(first);
    }
  };

  const handleValidate = () => {
    void datasetValidation.validate({
      path: datasetPath,
      remote: datasetRemote,
      use_case_slug: useCaseSlug,
      objective: objective as TrainingObjective,
    });
  };

  const handleSubmit = async () => {
    if (!rec.recommendation || !modelId.trim() || !datasetPath.trim()) return;
    await onSubmit({
      use_case_slug: useCaseSlug,
      objective: objective as TrainingObjective,
      model_id: modelId.trim(),
      dataset_path: datasetPath.trim(),
      dataset_remote: datasetRemote,
      output_dir: outputDir.trim(),
      gpu_indices: gpuIndices,
      tensorboard,
      recommendation: rec.recommendation,
    });
  };

  const canSubmit = Boolean(rec.recommendation) && modelId.trim() !== "" && datasetPath.trim() !== "";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <UseCasePicker useCases={useCases} value={useCaseSlug} onChange={setUseCaseSlug} />

      <ObjectivePicker useCase={activeUseCase} value={objective} onChange={setObjective} />

      <GpuSelector gpus={gpus} selected={gpuIndices} onChange={setGpuIndices} />

      <div>
        <Button variant="primary" loading={rec.loading} onClick={handleGenerate}>
          Generate recommendation
        </Button>
        {rec.error && (
          <div style={{ marginTop: "var(--space-3)" }}>
            <Callout tone="danger">{rec.error.message}</Callout>
          </div>
        )}
      </div>

      {rec.recommendation && (
        <Field label="Model" hint="or pick one from the candidates on the right">
          <Input value={modelId} onChange={(e) => setModelId(e.target.value)} />
        </Field>
      )}

      <DatasetField
        path={datasetPath}
        remote={datasetRemote}
        onPathChange={setDatasetPath}
        onRemoteChange={setDatasetRemote}
        onValidate={handleValidate}
        validation={datasetValidation.result}
        validating={datasetValidation.loading}
        validationError={datasetValidation.error?.message ?? null}
      />

      <Field label="Output directory">
        <Input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
      </Field>

      <Checkbox
        label="Write TensorBoard logs"
        checked={tensorboard}
        onChange={(e) => setTensorboard(e.target.checked)}
      />

      {submitError && <Callout tone="danger">{submitError}</Callout>}

      <div>
        <Button variant="primary" loading={submitting} disabled={!canSubmit} onClick={handleSubmit}>
          Start training
        </Button>
      </div>
    </div>
  );
}
