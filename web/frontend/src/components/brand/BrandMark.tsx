import { useId } from "react";
import { cn } from "../../lib/cn";
import "./BrandMark.css";

export interface BrandMarkProps {
  size?: number;
  variant?: "image" | "svg" | "animated";
  title?: string;
}

/** The M as circuit traces. Vertices used for solder pads. */
const M_PATH = "M22 80 L22 22 L50 54 L78 22 L78 80";
const VERTICES: Array<[number, number]> = [
  [22, 80],
  [22, 22],
  [78, 22],
  [78, 80],
];
const NODE: [number, number] = [50, 54];

export function BrandMark({ size = 28, variant = "svg", title = "ModelForge" }: BrandMarkProps) {
  const uid = useId().replace(/[:]/g, "");
  const emberId = `mfEmber-${uid}`;
  const glowId = `mfGlow-${uid}`;

  if (variant === "image") {
    return (
      <span className="mf-brandmark" style={{ width: size, height: size }}>
        <img
          className="mf-brandmark__img"
          src="/modelforge-mark.png"
          alt={title}
          width={size}
          height={size}
        />
      </span>
    );
  }

  const animated = variant === "animated";

  return (
    <span
      className={cn("mf-brandmark", animated && "mf-brandmark--animated")}
      style={{ width: size, height: size }}
    >
      <svg
        className="mf-brandmark__svg"
        width={size}
        height={size}
        viewBox="0 0 100 100"
        role="img"
        aria-label={title}
      >
        <defs>
          <linearGradient id={emberId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--ember-300)" />
            <stop offset="1" stopColor="var(--ember-600)" />
          </linearGradient>
          <filter id={glowId} x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>

        <rect
          x="3"
          y="3"
          width="94"
          height="94"
          rx="22"
          fill="var(--surface-3)"
          stroke="var(--border)"
          strokeWidth="1"
        />

        <path
          className="mf-brandmark__m"
          d={M_PATH}
          stroke={`url(#${emberId})`}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {VERTICES.map(([x, y]) => (
          <rect
            key={`${x}-${y}`}
            x={x - 3.5}
            y={y - 3.5}
            width="7"
            height="7"
            fill="var(--accent)"
          />
        ))}

        <circle
          cx={NODE[0]}
          cy={NODE[1]}
          r="6"
          fill="var(--accent)"
          filter={`url(#${glowId})`}
        />
        <circle
          className="mf-brandmark__node"
          cx={NODE[0]}
          cy={NODE[1]}
          r="4"
          fill="var(--accent-hover)"
        />
      </svg>
    </span>
  );
}
