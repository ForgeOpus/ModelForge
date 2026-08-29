import { Card } from "../../../components/ui";

const FEATURES = [
  {
    title: "Runs on your machine",
    body: "The trainer executes locally against your own GPUs — no data leaves the host.",
  },
  {
    title: "HF Hub model discovery",
    body: "Candidate base models are pulled from the Hub and filtered by task and license.",
  },
  {
    title: "Dataset validation",
    body: "Schema, split sizes and token statistics are checked before a run starts.",
  },
  {
    title: "TensorBoard logging",
    body: "Every run writes standard event files you can open in TensorBoard.",
  },
  {
    title: "Checkpoint resume",
    body: "Interrupted runs restart from the last saved checkpoint with optimizer state intact.",
  },
  {
    title: "Light & dark",
    body: "The console follows your system theme or an explicit toggle.",
  },
];

export function FeatureGrid() {
  return (
    <section className="lp-section">
      <div className="container">
        <div className="lp-head">
          <h2>Built for local iteration</h2>
        </div>
        <div className="lp-grid">
          {FEATURES.map((f) => (
            <Card key={f.title} interactive as="article">
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
