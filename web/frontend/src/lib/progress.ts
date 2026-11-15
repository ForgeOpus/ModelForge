import type { TrainingEvent } from "@/types";

export interface MetricPoint {
  step: number;
  loss?: number;
  grad_norm?: number;
  learning_rate?: number;
  mean_token_accuracy?: number;
  epoch?: number;
}

export interface RunProgress {
  step: number;
  maxSteps: number;
  epoch?: number;
  latest: MetricPoint | null;
  series: MetricPoint[];
  checkpoints: { step: number; path: string }[];
  terminal: "completed" | "cancelled" | "failed" | null;
  errorMessage: string | null;
  /** Estimated seconds remaining, from observed step rate. */
  etaSeconds: number | null;
}

/**
 * API timestamps are Unix seconds (`time.time()`). Fill a missing `ts` once,
 * from the client clock converted to seconds, so later math never mixes units.
 */
export function normalizeTrainingEvent(
  event: TrainingEvent,
  nowMs = Date.now(),
): TrainingEvent & { ts: number } {
  if (typeof event.ts === "number" && Number.isFinite(event.ts)) {
    return event as TrainingEvent & { ts: number };
  }
  return { ...event, ts: nowMs / 1000 };
}

/** Fold the raw event stream into everything the Run screen renders. */
export function deriveProgress(events: TrainingEvent[], nowMs = Date.now()): RunProgress {
  // One conversion at the boundary. The fold below treats `ts` as seconds.
  const normalized = events.map((event) => normalizeTrainingEvent(event, nowMs));
  const series: MetricPoint[] = [];
  const checkpoints: { step: number; path: string }[] = [];
  let step = 0;
  let maxSteps = 0;
  let epoch: number | undefined;
  let terminal: RunProgress["terminal"] = null;
  let errorMessage: string | null = null;
  let firstLogTs: number | null = null;
  let firstLogStep = 0;
  let lastLogTs: number | null = null;

  for (const event of normalized) {
    if (event.type === "log") {
      const p = event.payload ?? {};
      const logs = p.logs ?? {};
      if (typeof p.step === "number") step = Math.max(step, p.step);
      if (typeof p.max_steps === "number" && p.max_steps > 0) maxSteps = p.max_steps;
      if (typeof logs.epoch === "number") epoch = logs.epoch;

      const point: MetricPoint = { step: p.step ?? step };
      if (typeof logs.loss === "number") point.loss = logs.loss;
      if (typeof logs.grad_norm === "number") point.grad_norm = logs.grad_norm;
      if (typeof logs.learning_rate === "number") point.learning_rate = logs.learning_rate;
      if (typeof logs.mean_token_accuracy === "number") {
        point.mean_token_accuracy = logs.mean_token_accuracy;
      }
      if (typeof logs.epoch === "number") point.epoch = logs.epoch;
      // A final summary log carries no per-step loss; don't plot an empty point.
      if (Object.keys(point).length > 1) series.push(point);

      const ts = event.ts;
      if (firstLogTs === null) {
        firstLogTs = ts;
        firstLogStep = p.step ?? 0;
      }
      lastLogTs = ts;

    } else if (event.type === "checkpoint") {
      const p = event.payload ?? {};
      if (typeof p.step === "number" && typeof p.path === "string") {
        checkpoints.push({ step: p.step, path: p.path });
      }
    } else if (event.type === "completed" || event.type === "cancelled") {
      terminal = event.type;
    } else if (event.type === "error") {
      terminal = "failed";
      errorMessage = event.payload?.error ?? event.message ?? null;
    }
  }

  let etaSeconds: number | null = null;
  if (
    terminal === null &&
    firstLogTs !== null &&
    lastLogTs !== null &&
    maxSteps > 0 &&
    step > firstLogStep &&
    lastLogTs > firstLogTs
  ) {
    const stepsPerSecond = (step - firstLogStep) / (lastLogTs - firstLogTs);
    if (stepsPerSecond > 0) {
      // A step past max_steps is already finished; never report a negative ETA.
      etaSeconds = Math.max(0, (maxSteps - step) / stepsPerSecond);
    }
  }

  // The final log of a run is a summary that carries no per-step loss, so
  // taking the last point wholesale would blank the stat row exactly when the
  // run finishes. Carry the most recent defined value for each metric instead.
  const latest: MetricPoint | null = series.length
    ? series.reduce<MetricPoint>((acc, point) => {
        const next = { ...acc };
        for (const [key, value] of Object.entries(point)) {
          if (typeof value === "number") next[key as keyof MetricPoint] = value;
        }
        return next;
      }, { step })
    : null;

  return {
    step,
    maxSteps,
    epoch,
    latest,
    series,
    checkpoints,
    terminal,
    errorMessage,
    etaSeconds,
  };
}
