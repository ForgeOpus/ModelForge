import { useEffect, useRef, useState } from "react";
import { api, wsUrl } from "./api";
import type { TrainingEvent } from "@/types";

const MAX_EVENTS = 2000;

export type StreamState = "connecting" | "open" | "closed";

/* Live training events for one run.

   Two things this has to get right that the old UI did not:
   the buffer is seeded from the server's replay endpoint, so a page reload
   mid-run repaints the whole history instead of showing an empty log; and the
   socket reconnects with backoff, because a training process that saturates
   the GPU can starve the event loop long enough to drop the connection. */
export function useRunEvents(runId: number | null) {
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [state, setState] = useState<StreamState>("closed");
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const closedByUs = useRef(false);

  useEffect(() => {
    if (runId == null) {
      setEvents([]);
      setState("closed");
      return;
    }

    let cancelled = false;
    closedByUs.current = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const append = (event: TrainingEvent) => {
      if (event.type === "heartbeat") return;
      setEvents((prev) => {
        const next = [...prev, { ...event, ts: event.ts ?? Date.now() }];
        return next.length > MAX_EVENTS ? next.slice(next.length - MAX_EVENTS) : next;
      });
    };

    const connect = () => {
      if (cancelled) return;
      setState("connecting");
      const socket = new WebSocket(wsUrl(`/api/runs/${runId}/events`));
      socketRef.current = socket;

      socket.onopen = () => {
        if (cancelled) return;
        retryRef.current = 0;
        setState("open");
      };
      socket.onmessage = (message) => {
        try {
          append(JSON.parse(message.data) as TrainingEvent);
        } catch {
          /* a malformed frame should not kill the stream */
        }
      };
      socket.onclose = () => {
        if (cancelled || closedByUs.current) {
          setState("closed");
          return;
        }
        setState("closed");
        retryRef.current += 1;
        const delay = Math.min(1000 * 2 ** (retryRef.current - 1), 15000);
        timer = setTimeout(connect, delay);
      };
      socket.onerror = () => socket.close();
    };

    // Paint history first, then attach the live stream on top of it.
    api
      .runEvents(runId)
      .then((history) => {
        if (cancelled) return;
        setEvents(history.filter((e) => e.type !== "heartbeat"));
      })
      .catch(() => {
        /* replay is a convenience; the live stream still works without it */
      })
      .finally(() => {
        if (!cancelled) connect();
      });

    return () => {
      cancelled = true;
      closedByUs.current = true;
      if (timer) clearTimeout(timer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [runId]);

  return { events, state };
}

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

/** Fold the raw event stream into everything the Run screen renders. */
export function deriveProgress(events: TrainingEvent[]): RunProgress {
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

  for (const event of events) {
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

      const ts = event.ts ?? Date.now();
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
    const stepsPerSecond = (step - firstLogStep) / ((lastLogTs - firstLogTs) / 1000);
    if (stepsPerSecond > 0) etaSeconds = (maxSteps - step) / stepsPerSecond;
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
