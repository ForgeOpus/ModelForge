import { useEffect, useRef, useState } from "react";
import { animate as motionAnimate } from "motion";

/* Motion policy for ModelForge.

   This is instrumentation, not a brochure. The rule used throughout: motion
   either shows a value changing, or shows the machine doing work. Anything
   that would merely decorate is left out.

   One consequence worth stating: every figure on screen is a measurement, and
   measurements on real instruments sweep -- they do not teleport. So numbers
   tween. That is the single most-used piece of motion in the app, and the
   least showy. */

/** Live subscription to the user's motion preference, not a one-shot read. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const q = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(q.matches);
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return reduced;
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Tween a number toward its target so a readout sweeps rather than jumps.
 * Returns the animated value; snaps instantly under reduced motion, and on the
 * very first value so nothing counts up from zero on mount.
 */
export function useTween(target: number, duration = 0.6): number {
  const [value, setValue] = useState(target);
  const current = useRef(target);
  const primed = useRef(false);

  useEffect(() => {
    if (!Number.isFinite(target)) return;
    if (!primed.current || prefersReducedMotion()) {
      primed.current = true;
      current.current = target;
      setValue(target);
      return;
    }
    const from = current.current;
    if (from === target) return;
    const controls = motionAnimate(from, target, {
      duration,
      ease: [0.22, 0.8, 0.3, 1],
      onUpdate: (v) => {
        current.current = v;
        setValue(v);
      },
    });
    return () => controls.stop();
  }, [target, duration]);

  return value;
}

/** Shared easing and timings, so motion across the app reads as one system. */
export const EASE = [0.22, 0.8, 0.3, 1] as const;

export const TIMING = {
  /** Hover and press feedback. */
  tap: 0.12,
  /** Panels, popovers, route changes. */
  ui: 0.22,
  /** A value settling into place. */
  readout: 0.6,
  /** The plan-resolution sequence. */
  sequence: 1.1,
} as const;

/** Page/route transition, deliberately near-invisible. */
export const routeTransition = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -2 },
  transition: { duration: TIMING.ui, ease: EASE },
};

/**
 * Reveal props that collapse to nothing when motion is unwanted.
 *
 * Returning `{}` rather than a zero-duration animation matters: with no
 * `initial`, the element renders at its final state on the very first paint,
 * so the content never depends on an animation frame arriving to become
 * readable.
 */
export function reveal(
  reduced: boolean,
  { y = 12, delay = 0, duration = 0.5 } = {},
) {
  if (reduced) return {};
  return {
    initial: { opacity: 0, y },
    animate: { opacity: 1, y: 0 },
    transition: { duration, ease: EASE, delay },
  } as const;
}

/** Same, for scroll-triggered sections. */
export function revealOnScroll(
  reduced: boolean,
  { y = 14, amount = 0.15, duration = 0.5 } = {},
) {
  if (reduced) return {};
  return {
    initial: { opacity: 0, y },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, amount },
    transition: { duration, ease: EASE },
  } as const;
}

/** Children rise in sequence. Used for the rationale blocks only. */
export const stagger = (delay = 0.04) => ({
  animate: { transition: { staggerChildren: delay } },
});

export const riseItem = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: TIMING.ui, ease: EASE } },
};
