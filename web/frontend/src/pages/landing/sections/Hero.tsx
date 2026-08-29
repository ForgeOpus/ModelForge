import { Link } from "react-router-dom";
import { Badge, Card } from "../../../components/ui";
import { NeuralHero } from "../../../components/hero";

const FEATURES = [
  {
    title: "VRAM-aware",
    body: "Detects GPUs, VRAM, RAM and disk before it recommends anything.",
  },
  {
    title: "Explained decisions",
    body: "Every choice ships with the reasoning: the P_max math, the tier, the filter.",
  },
  {
    title: "PEFT tiers",
    body: "Full, LoRA bf16, QLoRA 8-bit, QLoRA 4-bit — resolved to your hardware.",
  },
  {
    title: "Streamed runs",
    body: "Live loss, steps and checkpoints over a WebSocket; resume from any checkpoint.",
  },
];

export function Hero() {
  return (
    <section className="lp-hero">
      <NeuralHero />
      <div className="container lp-hero__inner">
        <span className="lp-hero__chip tilt">
          <Badge color="var(--accent)">Recommendation engine + trainer</Badge>
        </span>

        <h1>
          Fine-tune the right base model for{" "}
          <span className="lp-hero__accent">the GPU you actually have.</span>
        </h1>

        <p className="lp-hero__sub container-narrow">
          ModelForge reads your VRAM, resolves your use case to a task and
          training strategy, picks a PEFT tier — full, LoRA bf16, QLoRA 8-bit or
          4-bit — with the parameter-budget math shown, chooses data-parallel or
          FSDP, then streams the run.
        </p>

        <div className="cluster lp-hero__ctas">
          <Link
            to="/app"
            className="ui-btn ui-btn--primary ui-btn--md shine-sweep"
          >
            Open Console
          </Link>
          <a href="#how-it-works" className="ui-btn ui-btn--secondary ui-btn--md">
            How it decides
          </a>
        </div>

        <div className="lp-hero__features">
          {FEATURES.map((f, i) => (
            <Card
              key={f.title}
              interactive
              className="fade-in-up"
              style={{ animationDelay: `${0.1 * i}s` }}
            >
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
