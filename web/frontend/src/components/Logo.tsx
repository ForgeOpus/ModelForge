import { cn } from "@/lib/utils";

/* Two marks, because one asset cannot do both jobs.

   The official mark is a 512px 3D render: black anodised metal, orange
   internal glow, circuit traces, and the wordmark set beneath it. It is
   gorgeous at size and unusable small -- it is raster, and its near-black
   ground (#0c0c0e) is baked in, so it cannot sit on a light surface without
   dragging a dark square along with it.

   So: the render is used as artwork, on its own plate where that dark ground
   is the point. For nav, favicons, and anything under ~48px there is a flat
   vector M traced from the render's own geometry -- square-cornered, stems at
   roughly a fifth of the width, valley descending to 54% height, measured off
   the original rather than invented. */

// Traced from the render: the M is very nearly square (measured aspect 1.04).
const M_PATH =
  "M3 28 L3 4 L9.5 4 L16 17 L22.5 4 L29 4 L29 28 L24 28 L24 12.5 " +
  "L17.8 24.5 L14.2 24.5 L8 12.5 L8 28 Z";

export function Mark({ className, duo = false }: { className?: string; duo?: boolean }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("shrink-0", className)} aria-hidden="true">
      {duo ? (
        <>
          {/* The render lights the M's left face hotter than its right. */}
          <defs>
            <linearGradient id="mf-mark-duo" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="var(--brand-hi)" />
              <stop offset="100%" stopColor="var(--brand)" />
            </linearGradient>
          </defs>
          <path d={M_PATH} fill="url(#mf-mark-duo)" />
        </>
      ) : (
        <path d={M_PATH} fill="currentColor" />
      )}
    </svg>
  );
}

/** Solid tile, for the favicon and anywhere the mark needs its own ground. */
export function MarkTile({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("shrink-0", className)} aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--brand)" />
      <g transform="translate(16 16) scale(0.7) translate(-16 -16)">
        <path d={M_PATH} fill="var(--brand-ink)" />
      </g>
    </svg>
  );
}

/** The official render. Only use it at a size where the detail earns its bytes. */
export function OfficialMark({
  className,
  alt = "ModelForge",
}: {
  className?: string;
  alt?: string;
}) {
  return (
    <img
      src="/favicon.jpg"
      alt={alt}
      width={512}
      height={512}
      className={cn("block select-none", className)}
      draggable={false}
    />
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "font-display font-bold uppercase leading-none tracking-[0.16em]",
        className,
      )}
    >
      ModelForge
    </span>
  );
}

export function Lockup({
  className,
  duo = false,
  size = "md",
}: {
  className?: string;
  duo?: boolean;
  size?: "sm" | "md" | "lg";
}) {
  const marks = { sm: "size-5", md: "size-6", lg: "size-9" };
  const words = { sm: "text-[13px]", md: "text-[15px]", lg: "text-[22px]" };
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <Mark className={marks[size]} duo={duo} />
      <Wordmark className={words[size]} />
    </span>
  );
}
