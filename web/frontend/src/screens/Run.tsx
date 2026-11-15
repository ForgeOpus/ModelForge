import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { toast } from "sonner";
import {
  Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer,
  Tooltip as RTooltip, XAxis, YAxis,
} from "recharts";
import { Activity, ExternalLink, Loader2, Square } from "lucide-react";
import { motion } from "motion/react";

import { api, ApiError } from "@/lib/api";
import { useStore } from "@/lib/store";
import { deriveProgress, useRunEvents, type MetricPoint } from "@/lib/ws";
import { tierColor } from "@/lib/tiers";
import { cn, formatDuration, parseDate } from "@/lib/utils";
import type { CheckpointRecord, RunRecord } from "@/types";

import { Button } from "@/components/ui/button";
import { Alert, EmptyState, Eyebrow, Panel } from "@/components/ui/misc";
import { StatusChip, TierChip } from "@/components/TierChip";
import { Readout } from "@/components/Readout";
import { EASE, prefersReducedMotion } from "@/lib/motion";

type Metric = "loss" | "grad_norm" | "learning_rate" | "mean_token_accuracy";

const METRICS: { key: Metric; label: string }[] = [
  { key: "loss", label: "loss" },
  { key: "grad_norm", label: "grad_norm" },
  { key: "learning_rate", label: "learning_rate" },
  { key: "mean_token_accuracy", label: "token_acc" },
];

export function Run() {
  const { runId } = useParams();
  const { runs, activeRun, refreshRuns } = useStore();

  // No run in the URL means "whatever is live", falling back to the newest.
  const targetId = runId ? Number(runId) : (activeRun?.id ?? runs[0]?.id ?? null);
  const run: RunRecord | null = useMemo(
    () => runs.find((r) => r.id === targetId) ?? null,
    [runs, targetId],
  );

  const { events, state } = useRunEvents(targetId);
  const progress = useMemo(() => deriveProgress(events), [events]);
  const [metric, setMetric] = useState<Metric>("loss");
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (targetId == null) return;
    api.checkpoints(targetId).then(setCheckpoints).catch(() => setCheckpoints([]));
  }, [targetId, progress.checkpoints.length]);

  // A terminal event means the ledger row just changed too.
  useEffect(() => {
    if (progress.terminal) void refreshRuns();
  }, [progress.terminal, refreshRuns]);

  if (targetId == null || !run) {
    return (
      <EmptyState
        icon={<Activity />}
        title="No runs yet"
        action={
          <Button variant="primary" asChild>
            <Link to="/app">Configure a fine-tune</Link>
          </Button>
        }
      >
        Once a run starts, its loss curve, checkpoints, and log stream appear here — and stay here
        after it finishes.
      </EmptyState>
    );
  }

  const color = tierColor(run.peft);
  const live = run.status === "running" || run.status === "queued";
  const pct = progress.maxSteps > 0 ? (progress.step / progress.maxSteps) * 100 : 0;
  const finishedAt = live ? Date.now() : parseDate(run.updated_at);
  const elapsed = (finishedAt - parseDate(run.created_at)) / 1000;

  async function cancel() {
    setCancelling(true);
    try {
      await api.cancelRun(run!.id);
      toast.success("Cancellation requested. Training stops after the current step.");
      await refreshRuns();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.friendly : String(err));
    } finally {
      setCancelling(false);
    }
  }

  async function resume(path: string) {
    try {
      const next = await api.resumeRun(run!.id, path);
      toast.success(`Resumed as run #${next.id}`);
      await refreshRuns();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.friendly : String(err));
    }
  }

  return (
    <div className="flex flex-col gap-5 overflow-auto px-6 py-5 pb-10">
      <div className="flex flex-wrap items-center gap-3.5">
        <StatusChip status={run.status} />
        <span className="font-mono text-[13px]">{run.model_id}</span>
        <TierChip peft={run.peft} />
        <div className="flex-1" />
        {run.tensorboard_url && (
          <Button variant="ghost" size="sm" asChild>
            <a href={run.tensorboard_url} target="_blank" rel="noreferrer">
              TensorBoard <ExternalLink />
            </a>
          </Button>
        )}
        {live && (
          <Button variant="danger" size="sm" onClick={cancel} disabled={cancelling}>
            {cancelling ? <Loader2 className="animate-spin" /> : <Square />}
            Cancel run
          </Button>
        )}
      </div>

      {run.status === "failed" && run.error && (
        <Alert
          tone={run.error.includes("stopped before this run") ? "warn" : "danger"}
          title={
            run.error.includes("stopped before this run")
              ? "Interrupted, not failed"
              : "Training failed"
          }
        >
          {run.error}
        </Alert>
      )}

      <Panel className="px-4 py-3.5">
        <div className="mb-2.5 flex flex-wrap gap-x-4 font-mono text-[11.5px] text-dim">
          <span>
            step <Readout value={progress.step} />
            {progress.maxSteps > 0 && ` / ${progress.maxSteps.toLocaleString()}`}
          </span>
          <span className="flex-1" />
          {progress.epoch != null && <span>epoch {progress.epoch.toFixed(2)}</span>}
          {live && progress.etaSeconds != null && (
            <span>{formatDuration(progress.etaSeconds)} left</span>
          )}
          {!live && <span>{progress.terminal ?? run.status}</span>}
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-track">
          <motion.div
            className="h-full rounded-full"
            style={{ background: color }}
            animate={{ width: `${Math.min(100, pct)}%` }}
            transition={{ duration: 0.7, ease: EASE }}
          />
        </div>
      </Panel>

      <dl className="grid grid-cols-[repeat(auto-fit,minmax(104px,1fr))] gap-px overflow-hidden rounded-md border border-line bg-line">
        <Stat label="Loss" num={progress.latest?.loss} digits={4} />
        <Stat label="Grad norm" num={progress.latest?.grad_norm} digits={2} />
        <Stat label="Learn rate" value={progress.latest?.learning_rate?.toExponential(1)} />
        <Stat
          label="Token acc"
          num={
            progress.latest?.mean_token_accuracy != null
              ? progress.latest.mean_token_accuracy * 100
              : undefined
          }
          digits={1}
          suffix="%"
        />
        <Stat label="Checkpoints" num={checkpoints.length} />
        <Stat label="Elapsed" value={formatDuration(elapsed)} />
      </dl>

      <Panel className="px-4 pb-2.5 pt-3.5">
        <header className="mb-3 flex flex-wrap items-center gap-2.5">
          <Eyebrow>Training {metric.replace(/_/g, " ")}</Eyebrow>
          {progress.checkpoints.length > 12 && (
            <span className="font-mono text-[10px] text-faint">
              {progress.checkpoints.length} checkpoints
            </span>
          )}
          <div className="flex-1" />
          <div className="flex flex-wrap gap-0.5">
            {METRICS.map((m) => (
              <button
                key={m.key}
                type="button"
                aria-pressed={metric === m.key}
                onClick={() => setMetric(m.key)}
                className={cn(
                  "rounded-xs border px-2 py-1 font-mono text-[10.5px] transition-colors",
                  metric === m.key
                    ? "border-line bg-surface text-text"
                    : "border-transparent text-faint hover:text-text",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </header>
        <LossChart
          series={progress.series}
          metric={metric}
          color={color}
          checkpoints={progress.checkpoints}
          maxSteps={progress.maxSteps}
          live={live}
        />
      </Panel>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_268px]">
        <Console events={events} streaming={state} live={live} />
        <Checkpoints items={checkpoints} onResume={resume} disabled={live} />
      </div>
    </div>
  );
}

function Stat({
  label, value, num, digits, suffix,
}: {
  label: string; value?: string; num?: number; digits?: number; suffix?: string;
}) {
  return (
    <div className="bg-ground px-3 py-2.5">
      <dt className="mb-1 text-[10px] uppercase tracking-[0.13em] text-faint">{label}</dt>
      <dd className="m-0 font-mono text-[15px] tnum">
        {num != null ? (
          <Readout value={num} digits={digits ?? 0} suffix={suffix} />
        ) : (
          (value ?? "—")
        )}
      </dd>
    </div>
  );
}

function LossChart({
  series,
  metric,
  color,
  checkpoints,
  maxSteps,
  live,
}: {
  series: MetricPoint[];
  metric: Metric;
  color: string;
  checkpoints: { step: number }[];
  maxSteps: number;
  live: boolean;
}) {
  const data = series.filter((p) => typeof p[metric] === "number");

  if (data.length === 0) {
    return (
      <div className="flex h-[190px] flex-col items-center justify-center gap-2.5 px-6 text-center text-[12.5px] text-faint">
        {live ? (
          <>
            <motion.span
              className="size-1.5 rounded-full"
              style={{ background: color }}
              animate={{ opacity: [1, 0.2, 1] }}
              transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
            />
            Waiting for the first logged step
          </>
        ) : (
          <p className="max-w-sm leading-relaxed">
            No metrics were recorded for this run. Runs finished before this version of ModelForge
            did not keep their history — new runs do.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="h-[190px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`fill-${metric}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.26} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis
            dataKey="step"
            type="number"
            domain={[0, maxSteps || "dataMax"]}
            tick={{ fill: "var(--faint)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--line)" }}
          />
          <YAxis
            tick={{ fill: "var(--faint)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            tickLine={false}
            axisLine={false}
            width={62}
            domain={["auto", "auto"]}
            tickFormatter={makeTickFormatter(data.map((d) => d[metric] as number))}
          />
          <RTooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              borderRadius: 5,
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              color: "var(--text)",
            }}
            labelStyle={{ color: "var(--faint)" }}
          />
          {/* Checkpoints are where you could resume from, so they belong on
              the same axis as the loss you'd be resuming into. */}
          {checkpoints.length <= 12 &&
            checkpoints.map((c) => (
              <ReferenceLine
                key={c.step}
                x={c.step}
                stroke="var(--tier-lora)"
                strokeWidth={1.5}
                strokeDasharray="2 3"
              />
            ))}
          <Area
            type="monotone"
            dataKey={metric}
            stroke={color}
            strokeWidth={1.75}
            fill={`url(#fill-${metric})`}
            dot={
              live
                ? // Only the leading point is drawn, and it breathes, so you can
                  // see at a glance that data is still arriving.
                  (props: { cx?: number; cy?: number; index?: number }) => {
                    const last = (props.index ?? -1) === data.length - 1;
                    if (!last || props.cx == null || props.cy == null) {
                      return <g key={props.index} />;
                    }
                    return (
                      <circle
                        key={props.index}
                        cx={props.cx}
                        cy={props.cy}
                        r={3}
                        fill={color}
                        stroke="var(--ground)"
                        strokeWidth={1.5}
                        className={prefersReducedMotion() ? undefined : "mf-live-dot"}
                      />
                    );
                  }
                : false
            }
            activeDot={{ r: 3.5, fill: color, stroke: "var(--ground)", strokeWidth: 2 }}
            isAnimationActive={!prefersReducedMotion()}
            animationDuration={700}
            animationEasing="ease-out"
          />

        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function Console({
  events,
  streaming,
  live,
}: {
  events: import("@/types").TrainingEvent[];
  streaming: string;
  live: boolean;
}) {
  const [filter, setFilter] = useState<"all" | "checkpoint" | "error">("all");
  const [follow, setFollow] = useState(true);
  const boxRef = useRef<HTMLDivElement>(null);

  const shown = events.filter((e) =>
    filter === "all"
      ? true
      : filter === "checkpoint"
        ? e.type === "checkpoint"
        : e.type === "error",
  );

  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [shown.length, follow]);

  return (
    <Panel className="overflow-hidden">
      <header className="flex items-center gap-2.5 border-b border-line px-3.5 py-2.5">
        <Eyebrow>Event log</Eyebrow>
        <div className="flex-1" />
        {live && (
          <span className="font-mono text-[10.5px] text-faint">
            {streaming === "open" ? "streaming" : streaming}
          </span>
        )}
        <label className="flex items-center gap-1.5 text-[11px] text-faint">
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => setFollow(e.target.checked)}
            className="accent-[var(--brand)]"
          />
          Follow
        </label>
        <div className="flex flex-wrap gap-0.5">
          {(["all", "checkpoint", "error"] as const).map((f) => (
            <button
              key={f}
              type="button"
              aria-pressed={filter === f}
              onClick={() => setFilter(f)}
              className={cn(
                "rounded-xs border px-2 py-0.5 font-mono text-[10.5px]",
                filter === f
                  ? "border-line bg-surface text-text"
                  : "border-transparent text-faint hover:text-text",
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </header>
      <div ref={boxRef} className="max-h-[260px] overflow-auto font-mono text-[11.5px]">
        {shown.length === 0 ? (
          <p className="px-3.5 py-6 text-center text-faint">
            {live ? "Nothing logged yet." : "No timeline was kept for this run."}
          </p>
        ) : (
          shown.map((e, i) => (
            <div key={i} className="flex gap-3 border-b border-line px-3.5 py-1.5 last:border-b-0">
              <time className="shrink-0 text-faint">
                {e.ts != null ? new Date(e.ts * 1000).toLocaleTimeString() : "—"}
              </time>
              <span
                className="w-[74px] shrink-0 text-[10px] uppercase tracking-wider"
                style={{
                  color:
                    e.type === "checkpoint"
                      ? "var(--tier-lora)"
                      : e.type === "error"
                        ? "var(--danger)"
                        : e.type === "status" || e.type === "completed"
                          ? "var(--ok)"
                          : "var(--faint)",
                }}
              >
                {e.type}
              </span>
              <span className="min-w-0 flex-1 break-words text-dim">{describe(e)}</span>
            </div>
          ))
        )}
      </div>
    </Panel>
  );
}

/** Axis labels: enough precision to separate the ticks, no more. */
function makeTickFormatter(values: number[]) {
  const finite = values.filter((v) => Number.isFinite(v));
  const span = finite.length ? Math.max(...finite) - Math.min(...finite) : 0;
  return (value: number): string => {
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(1);
    if (span === 0) return abs >= 1 ? value.toFixed(2) : value.toFixed(3);
    // Roughly five ticks; resolve one step finer than the gap between them.
    const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(span / 5)) + 1));
    return value.toFixed(decimals);
  };
}

/** Turn an event's payload into a line worth reading, rather than dumping it. */
function describe(e: import("@/types").TrainingEvent): string {
  if (e.type === "log" && e.payload?.logs) {
    const l = e.payload.logs;
    const parts: string[] = [];
    if (e.payload.step != null) parts.push(`step ${e.payload.step}`);
    if (typeof l.loss === "number") parts.push(`loss ${l.loss.toFixed(4)}`);
    if (typeof l.grad_norm === "number") parts.push(`grad_norm ${l.grad_norm.toFixed(2)}`);
    if (typeof l.learning_rate === "number") parts.push(`lr ${l.learning_rate.toExponential(1)}`);
    if (typeof l.epoch === "number") parts.push(`epoch ${l.epoch.toFixed(2)}`);
    if (parts.length) return parts.join(" · ");
  }
  if (e.type === "checkpoint" && e.payload?.path) return e.payload.path;
  return e.message || e.type;
}

function Checkpoints({
  items,
  onResume,
  disabled,
}: {
  items: CheckpointRecord[];
  onResume: (path: string) => void;
  disabled: boolean;
}) {
  return (
    <Panel className="self-start overflow-hidden">
      <header className="border-b border-line px-3.5 py-2.5">
        <Eyebrow>Checkpoints</Eyebrow>
      </header>
      {items.length === 0 ? (
        <p className="px-3.5 py-6 text-center text-[12px] text-faint">
          None saved yet. One is written every <code className="font-mono">save_steps</code> steps.
        </p>
      ) : (
        [...items]
          .sort((a, b) => (b.step ?? 0) - (a.step ?? 0))
          .map((c) => (
            <div key={c.id} className="flex items-center gap-2.5 border-b border-line px-3.5 py-2 last:border-b-0">
              <span className="min-w-0 flex-1">
                <span className="block font-mono text-[11.5px]">
                  step {c.step?.toLocaleString() ?? "?"}
                </span>
                <span className="mt-0.5 block truncate font-mono text-[10.5px] text-faint">
                  {c.path}
                </span>
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onResume(c.path)}
                disabled={disabled}
                title={disabled ? "Cancel the running job first" : "Start a new run from here"}
              >
                Resume
              </Button>
            </div>
          ))
      )}
    </Panel>
  );
}
