export function AboutPanel() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)", color: "var(--text-2)" }}>
      <p>
        ModelForge fine-tunes open language models against your own datasets without asking you to
        hand-pick a base model, a PEFT scheme, or a parallelism plan. It reads your hardware, matches
        it to your task, and launches a run that fits.
      </p>

      <a href="/app" style={{ color: "var(--accent)" }}>
        Open the app
      </a>

      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
        <strong style={{ color: "var(--text-1)" }}>How the recommendation engine decides</strong>
        <ol style={{ paddingLeft: "var(--space-5)" }}>
          <li>Detect hardware — CPU cores, RAM, and every GPU with its VRAM.</li>
          <li>Resolve the objective — supervised, preference, or reinforcement — from your use case.</li>
          <li>Budget parameters per PEFT tier — full fine-tune, LoRA, or QLoRA at 8-bit or 4-bit.</li>
          <li>Plan parallelism — data parallel or FSDP across the available GPUs.</li>
        </ol>
      </div>
    </div>
  );
}
