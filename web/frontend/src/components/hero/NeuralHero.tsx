import "./NeuralHero.css";

interface Node {
  x: number;
  y: number;
  big?: boolean;
}

const NODES: Node[] = [
  { x: 90, y: 130, big: true },
  { x: 250, y: 420 },
  { x: 340, y: 90 },
  { x: 470, y: 300, big: true },
  { x: 560, y: 560 },
  { x: 660, y: 140 },
  { x: 740, y: 400 },
  { x: 850, y: 240, big: true },
  { x: 960, y: 520 },
  { x: 1050, y: 110 },
  { x: 1120, y: 360 },
];

const EDGES: Array<[number, number]> = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 3],
  [2, 5],
  [3, 4],
  [3, 6],
  [4, 8],
  [5, 6],
  [5, 9],
  [6, 7],
  [7, 8],
  [7, 10],
  [9, 10],
  [8, 10],
];

/** Decorative animated neural-network background for the hero section. */
export function NeuralHero() {
  return (
    <svg
      className="neural-hero"
      viewBox="0 0 1200 700"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <filter id="heroGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {EDGES.map(([a, b], i) => {
        const n1 = NODES[a];
        const n2 = NODES[b];
        return (
          <g key={`edge-${a}-${b}`}>
            <line
              x1={n1.x}
              y1={n1.y}
              x2={n2.x}
              y2={n2.y}
              stroke="var(--border-strong)"
              strokeWidth="1"
            />
            <line
              className="neural-hero__flow"
              x1={n1.x}
              y1={n1.y}
              x2={n2.x}
              y2={n2.y}
              stroke="var(--accent)"
              strokeWidth="1.5"
              strokeDasharray="6 180"
              style={{ animationDelay: `${i * -0.6}s` }}
            />
          </g>
        );
      })}

      {NODES.map((n, i) => (
        <circle
          key={`node-${i}`}
          className="neural-hero__node"
          cx={n.x}
          cy={n.y}
          r={n.big ? 6 : 4}
          fill="var(--accent)"
          filter="url(#heroGlow)"
          style={{ animationDelay: `${i * 0.27}s` }}
        />
      ))}
    </svg>
  );
}
