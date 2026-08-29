import { Card } from "../../../components/ui";

const STEPS = [
  {
    title: "Detect hardware",
    body: "Physical cores, RAM, free disk, and every visible GPU with its VRAM.",
  },
  {
    title: "Resolve the objective",
    body: "Your use case + objective map to a FinetuningTask and a training strategy slug.",
  },
  {
    title: "Budget the parameters",
    body: "For each PEFT tier it computes P_max from available VRAM and picks the richest tier that fits.",
  },
  {
    title: "Plan parallelism & launch",
    body: "Data-parallel or FSDP based on GPU count and per-GPU VRAM, then the trainer streams events.",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="lp-section">
      <div className="container">
        <div className="lp-head">
          <h2>How it decides</h2>
        </div>
        <div className="lp-steps">
          {STEPS.map((s, i) => (
            <Card key={s.title} interactive as="article" className="lp-step">
              <span className="lp-step__num">{i + 1}</span>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
