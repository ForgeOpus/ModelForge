import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { deriveProgress, normalizeTrainingEvent } from "./progress.ts";
import type { TrainingEvent } from "../types.ts";

const T0 = 1_700_000_000;

function log(step: number, ts: number | undefined, maxSteps = 2000): TrainingEvent {
  const event: TrainingEvent = {
    run_id: 1,
    type: "log",
    message: `step ${step}`,
    payload: { step, max_steps: maxSteps, logs: { loss: 1.5 } },
  };
  if (ts !== undefined) event.ts = ts;
  return event;
}

describe("deriveProgress ETA", () => {
  it("uses backend unix seconds without treating them as milliseconds", () => {
    // 1000 steps over 1000 seconds is 1 step/s, so 1000 steps remain.
    // The old ms formula would report 1 second remaining.
    const progress = deriveProgress([
      log(0, T0),
      log(1000, T0 + 1000),
    ]);

    const stepsPerSecond = (1000 - 0) / 1000;
    assert.equal(progress.step, 1000);
    assert.equal(progress.maxSteps, 2000);
    assert.equal(stepsPerSecond, 1);
    assert.equal(progress.etaSeconds, (2000 - 1000) / stepsPerSecond);
  });

  it("normalizes a missing timestamp to seconds before estimating", () => {
    const nowMs = (T0 + 1000) * 1000;
    const missing = log(1000, undefined);
    const normalized = normalizeTrainingEvent(missing, nowMs);

    assert.equal(normalized.ts, T0 + 1000);
    assert.equal(
      deriveProgress([log(0, T0), missing], nowMs).etaSeconds,
      1000,
    );
  });

  it("clamps an overrun or completed estimate to zero", () => {
    const overrun = deriveProgress([
      log(0, T0, 100),
      log(150, T0 + 10, 100),
    ]);
    const completed = deriveProgress([
      log(0, T0, 100),
      log(100, T0 + 10, 100),
    ]);

    assert.equal(overrun.etaSeconds, 0);
    assert.equal(completed.etaSeconds, 0);
    assert.ok((overrun.etaSeconds ?? 0) >= 0);
  });

  it("does not invent a rate when every event lacks a timestamp", () => {
    const nowMs = (T0 + 5) * 1000;
    const progress = deriveProgress([log(10, undefined), log(20, undefined)], nowMs);

    assert.equal(progress.etaSeconds, null);
    assert.equal(normalizeTrainingEvent(log(10, undefined), nowMs).ts, nowMs / 1000);
  });
});
