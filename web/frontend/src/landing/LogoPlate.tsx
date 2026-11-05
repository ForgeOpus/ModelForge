import { motion } from "motion/react";
import { OfficialMark } from "@/components/Logo";
import { EASE, reveal, useReducedMotion } from "@/lib/motion";

/* The official mark, presented as an object rather than pasted onto the page.

   The render carries its own near-black ground (#0c0c0e) baked into the JPEG,
   so it can never float cleanly on a light surface. Rather than fight that
   with a lossy background knock-out, the plate makes the dark ground
   deliberate: a lit panel the mark sits inside, which holds in either theme.

   The bloom behind it is sampled from the mark's own internal glow. It breathes
   once on load and then stays still -- the logo is not the story, the budget
   meter beside it is. */
export function LogoPlate({ className }: { className?: string }) {
  const reduced = useReducedMotion();
  return (
    <motion.div className={className} {...reveal(reduced, { y: 10, duration: 0.55 })}>
      <div
        className="relative overflow-hidden rounded-lg border"
        style={{ borderColor: "#242428", background: "#0c0c0e" }}
      >
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute left-1/2 top-1/2 size-[62%] -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            background: "radial-gradient(circle, #d9580544 0%, transparent 68%)",
            filter: "blur(26px)",
          }}
          initial={reduced ? false : { opacity: 0, scale: 0.75 }}
          animate={reduced ? { opacity: 0.62 } : { opacity: [0, 1, 0.62], scale: [0.75, 1.12, 1] }}
          transition={reduced ? { duration: 0 } : { duration: 1.5, ease: EASE, delay: 0.2, times: [0, 0.55, 1] }}
        />
        <OfficialMark className="relative w-full" />
      </div>
    </motion.div>
  );
}
