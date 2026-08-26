import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Check, ChevronDown, Loader2, RotateCcw } from "lucide-react";

import { api, ApiError } from "@/lib/api";
import { useStore } from "@/lib/store";
import { footprint, gpuVram, rawVram } from "@/lib/vram";
import { cn } from "@/lib/utils";
import { formatCount, formatParams } from "@/lib/utils";
import type { DatasetValidation, Recommendation, UseCase } from "@/types";

import { Button } from "@/components/ui/button";
import { Input, Label, Segmented, Toggle } from "@/components/ui/field";
import { Alert, Chip, Eyebrow, Panel } from "@/components/ui/misc";
import { VramMeter, VramMini } from "@/components/VramMeter";
import { PlanBuilding } from "@/components/PlanBuilding";
import { motion } from "motion/react";
import { EASE, TIMING, riseItem, stagger } from "@/lib/motion";
import { AdvancedPanel, type Overrides } from "./ConfigureAdvanced";

type StepState = "done" | "todo" | "attn";

/* The gutter encodes *state*, not order.

   Order is already obvious from position on the page; whether a step is
   satisfied is not, and that's what someone scanning this screen needs. */
function Gutter({ state, last }: { state: StepState; last?: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1.5 pt-0.5">
      <span
        className={cn(
          "grid size-3.5 shrink-0 place-items-center rounded-full border-[1.5px]",
          state === "done" && "border-ok bg-ok",
          state === "attn" && "border-warn",
          state === "todo" && "border-line-strong",
        )}
      >
        {state === "done" && <Check className="size-2.5 text-ground" strokeWidth={3.5} />}
        {state === "attn" && <span className="size-1 rounded-full bg-warn" />}
      </span>
      {!last && <span className="w-px flex-1 bg-line" />}
    </div>
  );
}

function Step({
  state,
  title,
  hint,
  last,
  children,
}: {
  state: StepState;
  title: string;
  hint?: string;
  last?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className="grid grid-cols-[20px_minmax(0,1fr)] gap-3.5">
      <Gutter state={state} last={last} />
      <div className="min-w-0 pb-1">
        <h3 className="eyebrow mb-3">
          {title}
          {hint && (
            <em className="ml-2.5 text-[12px] normal-case not-italic tracking-normal text-dim">
              {hint}
            </em>
          )}
        </h3>
        {children}
      </div>
    </section>
  );
}

export function Configure() {
  const { system, hardware, activeRun, refreshRuns } = useStore();
  const navigate = useNavigate();

  const [useCaseSlug, setUseCaseSlug] = useState("instruction_following");
  const [objective, setObjective] = useState("supervised");
  const [modelId, setModelId] = useState("");
  const [customOpen, setCustomOpen] = useState(false);
  const [customId, setCustomId] = useState("");
  const [customCheck, setCustomCheck] = useState<{ busy: boolean; error: string | null }>({
    busy: false,
    error: null,
  });

  const [datasetPath, setDatasetPath] = useState("");
  const [datasetRemote, setDatasetRemote] = useState(false);
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [validating, setValidating] = useState(false);

  const [outputDir, setOutputDir] = useState("./modelforge_output");
  const [gpuIndices, setGpuIndices] = useState<number[]>([]);
  const [tensorboard, setTensorboard] = useState(true);

  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planSerial, setPlanSerial] = useState(0);
  const [planError, setPlanError] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Overrides>({});
  const [starting, setStarting] = useState(false);

  const useCase: UseCase | undefined = useMemo(
    () => system?.use_cases?.find((u) => u.slug === useCaseSlug),
    [system, useCaseSlug],
  );

  // Objectives are constrained by the use case; keep the pair valid.
  useEffect(() => {
    if (useCase && !useCase.compatible_objectives.includes(objective)) {
      setObjective(useCase.compatible_objectives[0] ?? "supervised");
    }
  }, [useCase, objective]);

  // Any change to the goal invalidates the plan that was built from it.
  useEffect(() => {
    setRecommendation(null);
    setOverrides({});
    setModelId("");
  }, [useCaseSlug, objective]);

  useEffect(() => {
    setValidation(null);
  }, [datasetPath, datasetRemote, useCaseSlug, objective]);

  const buildPlan = useCallback(async () => {
    setPlanning(true);
    setPlanError(null);
    try {
      const rec = await api.recommend({
        use_case_slug: useCaseSlug,
        objective,
        gpu_indices: gpuIndices.length ? gpuIndices : null,
        n: 5,
      });
      setRecommendation(rec);
      setPlanSerial((n) => n + 1);
      setOverrides({});
      if (!modelId) setModelId(rec.recommended_models[0]?.model_id ?? "");
    } catch (err) {
      setPlanError(err instanceof ApiError ? err.friendly : String(err));
    } finally {
      setPlanning(false);
    }
  }, [useCaseSlug, objective, gpuIndices, modelId]);

  const validateDataset = useCallback(async () => {
    if (!datasetPath.trim()) return;
    setValidating(true);
    try {
      setValidation(
        await api.validateDataset({
          path: datasetPath.trim(),
          remote: datasetRemote,
          use_case_slug: useCaseSlug,
          objective,
        }),
      );
    } catch (err) {
      setValidation({
        valid: false,
        path: datasetPath,
        remote: datasetRemote,
        columns: [],
        format_description: null,
        error: err instanceof ApiError ? err.friendly : String(err),
      });
    } finally {
      setValidating(false);
    }
  }, [datasetPath, datasetRemote, useCaseSlug, objective]);

  const checkCustomModel = useCallback(async () => {
    const id = customId.trim();
    if (!id) return;
    setCustomCheck({ busy: true, error: null });
    try {
      const result = await api.validateModelId(id, recommendation?.task_slug);
      if (result.valid) {
        setModelId(id);
        setCustomCheck({ busy: false, error: null });
        toast.success(`Using ${id}`);
      } else {
        setCustomCheck({ busy: false, error: result.error ?? "That model could not be loaded." });
      }
    } catch (err) {
      setCustomCheck({
        busy: false,
        error: err instanceof ApiError ? err.friendly : String(err),
      });
    }
  }, [customId, recommendation]);

  const totalVram = rawVram(hardware);
  // The plan's own reading is authoritative; detected VRAM is the fallback.
  const budgetGb = recommendation?.vram_estimate?.raw_vram_gb || totalVram || 1;
  const selectedModel = recommendation?.recommended_models.find((m) => m.model_id === modelId);
  // A custom model has no parameter count until the server reports one, so the
  // meter falls back to the plan's own ceiling rather than guessing.
  const meterParams =
    selectedModel?.num_parameters ??
    (recommendation?.vram_estimate?.max_params_by_tier[recommendation.peft_approach] ?? 0);

  const goalState: StepState = useCase ? "done" : "todo";
  const modelState: StepState = modelId ? "done" : recommendation ? "attn" : "todo";
  const dataState: StepState = validation?.valid
    ? "done"
    : validation && !validation.valid
      ? "attn"
      : datasetPath
        ? "attn"
        : "todo";
  const outputState: StepState = outputDir.trim() ? "done" : "attn";

  const blockedReason = activeRun
    ? `Run #${activeRun.id} is still going. Cancel it first.`
    : !recommendation
      ? "Build a plan first."
      : !modelId
        ? "Choose a base model."
        : !datasetPath.trim()
          ? "Point at a dataset."
          : !outputDir.trim()
            ? "Choose an output directory."
            : validation && !validation.valid
              ? "The dataset did not validate."
              : null;

  async function start() {
    if (!recommendation || blockedReason) return;
    setStarting(true);
    try {
      const merged = applyOverrides(recommendation, overrides);
      const run = await api.startRun({
        use_case_slug: useCaseSlug,
        objective,
        model_id: modelId,
        dataset_path: datasetPath.trim(),
        dataset_remote: datasetRemote,
        output_dir: outputDir.trim(),
        gpu_indices: gpuIndices,
        tensorboard,
        recommendation: merged,
      });
      await refreshRuns();
      toast.success(`Run #${run.id} started`);
      navigate(`/app/run/${run.id}`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.friendly : String(err));
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="grid h-full grid-cols-1 xl:grid-cols-[minmax(0,1fr)_348px]">
      <div className="flex flex-col gap-6 overflow-auto px-6 py-6 pb-12">
        <Step state={goalState} title="Goal" hint="What should the model learn to do?">
          <div className="grid grid-cols-[repeat(auto-fit,minmax(168px,1fr))] gap-2">
            {system?.use_cases?.map((u) => (
              <button
                key={u.slug}
                type="button"
                aria-pressed={u.slug === useCaseSlug}
                onClick={() => setUseCaseSlug(u.slug)}
                className={cn(
                  "rounded-sm border p-3 text-left transition-colors",
                  u.slug === useCaseSlug
                    ? "border-brand bg-raised"
                    : "border-line bg-surface hover:border-line-strong",
                )}
              >
                <b className="mb-1 block text-[13px] font-medium text-text">{u.name}</b>
                <span className="block text-[11.5px] leading-snug text-faint">
                  {u.description}
                </span>
              </button>
            ))}
            {!system && <div className="h-20 animate-pulse rounded-sm bg-raised" />}
          </div>

          {useCase && useCase.compatible_objectives.length > 0 && (
            <div className="mt-3">
              <Segmented
                label="Training objective"
                value={objective}
                onChange={setObjective}
                options={useCase.compatible_objectives.map((o) => ({ value: o, label: o }))}
              />
            </div>
          )}
        </Step>

        <Step
          state={modelState}
          title="Base model"
          hint={
            recommendation
              ? `${recommendation.recommended_models.length} candidates fit your budget`
              : "Build a plan to see what fits"
          }
        >
          {!recommendation ? (
            <div>
              <Button variant="primary" onClick={buildPlan} disabled={planning}>
                {planning && <Loader2 className="animate-spin" />}
                {planning ? "Reading your hardware…" : "Build a plan"}
              </Button>
              {planError && (
                <div className="mt-3">
                  <Alert tone="danger" title="Could not build a plan">
                    {planError}
                  </Alert>
                </div>
              )}
            </div>
          ) : recommendation.recommended_models.length === 0 ? (
            <Alert tone="warn" title="No models matched">
              The Hub returned nothing for this task within your parameter ceiling. Enter a model ID
              below, or free up VRAM and build the plan again.
            </Alert>
          ) : (
            <div className="overflow-hidden rounded-sm border border-line">
              {recommendation.recommended_models.map((m, i) => (
                <button
                  key={m.model_id}
                  type="button"
                  aria-pressed={m.model_id === modelId}
                  onClick={() => setModelId(m.model_id)}
                  className={cn(
                    "grid w-full grid-cols-[15px_minmax(0,1fr)_66px_74px_78px] items-center gap-3 px-3 py-2.5 text-left transition-colors",
                    i > 0 && "border-t border-line",
                    m.model_id === modelId ? "bg-raised" : "bg-surface hover:bg-raised",
                  )}
                >
                  <span
                    className={cn(
                      "size-2.5 rounded-full border-[1.5px]",
                      m.model_id === modelId ? "border-[3.5px] border-brand" : "border-line-strong",
                    )}
                  />
                  <span className="min-w-0">
                    <span className="block truncate font-mono text-[12px]">{m.model_id}</span>
                    <VramMini
                      className="mt-1.5 ring-1 ring-line"
                      numParams={m.num_parameters}
                      peft={recommendation.peft_approach}
                      totalGb={budgetGb}
                    />
                  </span>
                  <span className="text-right font-mono text-[11.5px] text-dim tnum">
                    {formatParams(m.num_parameters)}
                    <s className="block text-[10.5px] text-faint no-underline">params</s>
                  </span>
                  {(() => {
                    const fp = footprint(m.num_parameters, recommendation.peft_approach, budgetGb);
                    return (
                      <span
                        className="text-right font-mono text-[11.5px] tnum"
                        style={{ color: fp.over ? "var(--danger)" : "var(--text)" }}
                      >
                        {fp.usedGb.toFixed(1)} GB
                        {fp.over && (
                          <s className="block text-[10.5px] text-danger no-underline">
                            over budget
                          </s>
                        )}
                      </span>
                    );
                  })()}
                  <span className="text-right font-mono text-[11.5px] text-dim tnum">
                    {formatCount(m.downloads)}
                    <s className="block text-[10.5px] text-faint no-underline">pulls</s>
                  </span>
                </button>
              ))}
            </div>
          )}

          {recommendation && (
            <div className="mt-2">
              <button
                type="button"
                onClick={() => setCustomOpen((v) => !v)}
                className="flex items-center gap-1.5 py-1.5 text-[12px] text-dim hover:text-text"
              >
                <ChevronDown className={cn("size-3 transition-transform", customOpen && "rotate-180")} />
                Use a different model ID
              </button>
              {customOpen && (
                <div className="pt-1">
                  <div className="flex gap-2">
                    <Input
                      value={customId}
                      onChange={(e) => setCustomId(e.target.value)}
                      placeholder="org/model"
                      aria-label="Custom model ID"
                    />
                    <Button onClick={checkCustomModel} disabled={customCheck.busy}>
                      {customCheck.busy && <Loader2 className="animate-spin" />}
                      Check
                    </Button>
                  </div>
                  <p className="mt-1.5 text-[11.5px] text-faint">
                    Anything on the Hub that matches this task. Larger models than the plan allows
                    will train, but may run out of memory.
                  </p>
                  {customCheck.error && (
                    <div className="mt-2">
                      <Alert tone="danger">{customCheck.error}</Alert>
                    </div>
                  )}
                </div>
              )}
              {modelId && !recommendation.recommended_models.some((m) => m.model_id === modelId) && (
                <p className="mt-1.5 font-mono text-[11.5px] text-brand">using {modelId}</p>
              )}
            </div>
          )}
        </Step>

        <Step
          state={dataState}
          title="Training data"
          hint={useCase?.data_formats?.[objective]?.user_facing_description}
        >
          <div className="flex gap-2">
            <Input
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && validateDataset()}
              placeholder={datasetRemote ? "org/dataset" : "./data/train.jsonl"}
              aria-label="Dataset path or Hugging Face dataset ID"
            />
            <Button onClick={validateDataset} disabled={validating || !datasetPath.trim()}>
              {validating && <Loader2 className="animate-spin" />}
              Validate
            </Button>
          </div>

          <label className="mt-2.5 flex items-center gap-2 text-[12px] text-dim">
            <Toggle checked={datasetRemote} onChange={setDatasetRemote} label="Hugging Face dataset" />
            Load from the Hugging Face Hub instead of this machine
          </label>

          {!validation && (
            <p className="mt-2.5 text-[11.5px] leading-relaxed text-faint">
              Parquet, Arrow, JSONL, JSON, CSV, and TXT are read directly. Point at a folder and
              every matching file is loaded together.
            </p>
          )}

          {validation?.valid && (
            <>
              <p className="mt-2.5 flex items-center gap-1.5 text-[12px] text-ok">
                <Check className="size-3.5" strokeWidth={3} />
                {validation.row_count != null
                  ? `${validation.row_count.toLocaleString()} rows · columns matched`
                  : "Streaming dataset · columns not checked"}
              </p>
              {validation.columns.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {validation.columns.map((c) => (
                    <Chip key={c} className="border-line text-dim">
                      {c}
                    </Chip>
                  ))}
                </div>
              )}
            </>
          )}

          {validation && !validation.valid && (
            <div className="mt-2.5">
              <Alert tone="danger" title="This dataset will not train">
                {validation.error}
                {validation.format_description && (
                  <p className="mt-1.5">Expected: {validation.format_description}</p>
                )}
              </Alert>
            </div>
          )}
        </Step>

        <Step state={outputState} title="Output" hint="Where the trained adapter is written" last>
          <Input
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            aria-label="Output directory"
          />

          {hardware && hardware.gpus.length > 0 && (
            <div className="mt-3">
              <Label>GPUs to use</Label>
              <div className="flex flex-wrap gap-2">
                {hardware.gpus.map((gpu, i) => {
                  const on = gpuIndices.length === 0 || gpuIndices.includes(i);
                  const vram = gpuVram(gpu);
                  return (
                    <button
                      key={i}
                      type="button"
                      aria-pressed={on}
                      onClick={() =>
                        setGpuIndices((prev) => {
                          const all = hardware.gpus.map((_, idx) => idx);
                          const current = prev.length === 0 ? all : prev;
                          const next = current.includes(i)
                            ? current.filter((x) => x !== i)
                            : [...current, i].sort((a, b) => a - b);
                          // Empty means "all", which is what the API expects.
                          return next.length === all.length ? [] : next;
                        })
                      }
                      className={cn(
                        "rounded-sm border px-2.5 py-1.5 font-mono text-[11.5px] transition-colors",
                        on ? "border-brand text-text" : "border-line text-faint",
                      )}
                    >
                      {i}: {gpu.name} {vram != null ? `· ${vram.toFixed(0)} GB` : "· unified"}
                    </button>
                  );
                })}
              </div>
              {hardware.gpus.length > 1 && (
                <p className="mt-1.5 text-[11.5px] text-faint">
                  VRAM is summed across the GPUs you pick. Sharding is not configured for you.
                </p>
              )}
            </div>
          )}

          <label className="mt-3 flex items-center gap-2 text-[12px] text-dim">
            <Toggle checked={tensorboard} onChange={setTensorboard} label="Write TensorBoard logs" />
            Write TensorBoard logs alongside the output
          </label>

          {recommendation && (
            <AdvancedPanel
              recommendation={recommendation}
              overrides={overrides}
              onChange={setOverrides}
            />
          )}
        </Step>
      </div>

      <PlanColumn
        recommendation={recommendation}
        totalVram={totalVram}
        meterParams={meterParams}
        overrides={overrides}
        onResetOverrides={() => setOverrides({})}
        blockedReason={blockedReason}
        starting={starting}
        onStart={start}
        onRebuild={buildPlan}
        planning={planning}
        planSerial={planSerial}
      />
    </div>
  );
}

function PlanColumn({
  recommendation,
  totalVram,
  meterParams,
  overrides,
  onResetOverrides,
  blockedReason,
  starting,
  onStart,
  onRebuild,
  planning,
  planSerial,
}: {
  recommendation: Recommendation | null;
  totalVram: number;
  meterParams: number;
  overrides: Overrides;
  onResetOverrides: () => void;
  blockedReason: string | null;
  starting: boolean;
  onStart: () => void;
  onRebuild: () => void;
  planning: boolean;
  planSerial: number;
}) {
  const overrideCount = Object.keys(overrides).length;

  return (
    <aside className="flex flex-col gap-5 overflow-auto border-line bg-ground p-5 xl:border-l">
      <div>
        <Eyebrow className="mb-3.5">Plan</Eyebrow>
        {planning ? (
          <PlanBuilding />
        ) : recommendation ? (
          <Panel className="border-line bg-surface p-4">
            <VramMeter
              numParams={meterParams}
              peft={recommendation.peft_approach}
              totalGb={recommendation.vram_estimate?.raw_vram_gb || totalVram || 1}
              measureKey={planSerial}
            />
          </Panel>
        ) : (
          <Panel className="bg-surface p-4">
            <p className="text-[12.5px] leading-relaxed text-dim">
              ModelForge reads your hardware, then picks a model and a training recipe that fits
              inside it. Build a plan to see the budget.
            </p>
          </Panel>
        )}
      </div>

      {recommendation && (
        <>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3.5 gap-y-1.5">
            <Fact label="Task" value={recommendation.task_slug} />
            <Fact label="Strategy" value={recommendation.strategy_slug} />
            <Fact label="Provider" value={recommendation.provider_slug} />
            {recommendation.peft_config && (
              <>
                <Fact
                  label="LoRA"
                  value={`r=${recommendation.peft_config.r} · α=${recommendation.peft_config.lora_alpha}`}
                />
                <Fact
                  label="Targets"
                  value={recommendation.peft_config.target_modules.join(", ")}
                />
              </>
            )}
          </dl>

          {overrideCount > 0 && (
            <button
              type="button"
              onClick={onResetOverrides}
              className="flex items-center gap-1.5 self-start text-[11.5px] text-warn hover:underline"
            >
              <RotateCcw className="size-3" />
              {overrideCount} value{overrideCount === 1 ? "" : "s"} changed from the plan — reset
            </button>
          )}

          {/* The engine explains itself in four fields. Every other tool hides
              this; here it is the reason to trust the plan. */}
          <motion.div
            key={planSerial}
            className="flex flex-col gap-3.5"
            initial="initial"
            animate="animate"
            variants={stagger(0.06).animate ? { animate: { transition: { staggerChildren: 0.06, delayChildren: 0.15 } } } : undefined}
          >
            <Eyebrow>Why this plan</Eyebrow>
            <Reason label="Hardware" text={recommendation.rationale.vram_summary} />
            <Reason label="Routing" text={recommendation.rationale.use_case_resolution} />
            <Reason label="Precision" text={recommendation.rationale.peft_decision} />
            <Reason label="Candidates" text={recommendation.rationale.model_filter} />
          </motion.div>

          {recommendation.rationale.warnings.length > 0 && (
            <div className="flex flex-col gap-2">
              {recommendation.rationale.warnings.map((w, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: TIMING.ui, ease: EASE, delay: 0.42 + i * 0.06 }}
                >
                  <Alert tone="warn">{w}</Alert>
                </motion.div>
              ))}
            </div>
          )}

          <div className="mt-auto pt-2">
            <Button
              variant="primary"
              size="lg"
              className="w-full"
              onClick={onStart}
              disabled={!!blockedReason || starting}
            >
              {starting && <Loader2 className="animate-spin" />}
              Start training
            </Button>
            {blockedReason ? (
              <p className="mt-2 text-center text-[11.5px] text-faint">{blockedReason}</p>
            ) : (
              <button
                type="button"
                onClick={onRebuild}
                disabled={planning}
                className="mt-2 w-full text-center text-[11.5px] text-faint hover:text-text"
              >
                Rebuild the plan
              </button>
            )}
          </div>
        </>
      )}
    </aside>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-[11px] text-faint">{label}</dt>
      <dd className="m-0 truncate font-mono text-[11.5px] text-text">{value}</dd>
    </>
  );
}

/* The rationale arrives as engine prose with newlines and a per-tier table.
   Render it as text, not as a JSON blob. */
function Reason({ label, text }: { label: string; text: string }) {
  return (
    <motion.article variants={riseItem}>
      <h4 className="eyebrow mb-1 text-[10px]">{label}</h4>
      <p className="m-0 whitespace-pre-wrap text-[12px] leading-relaxed text-dim">{text}</p>
    </motion.article>
  );
}

/** Fold Advanced-panel edits back into the recommendation before it is POSTed.
    The API accepts the whole object verbatim, which is what makes every knob
    reachable without new endpoints. */
function applyOverrides(rec: Recommendation, overrides: Overrides): Recommendation {
  if (!Object.keys(overrides).length) return rec;
  const strategy_config = { ...(rec.strategy_config ?? {}) };
  let peft_config = rec.peft_config ? { ...rec.peft_config } : null;

  for (const [key, value] of Object.entries(overrides)) {
    if (key.startsWith("peft.")) {
      if (peft_config) {
        peft_config = { ...peft_config, [key.slice(5)]: value } as typeof peft_config;
      }
    } else {
      strategy_config[key] = value;
    }
  }
  return { ...rec, strategy_config, peft_config };
}
