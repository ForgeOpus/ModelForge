import { Boxes } from "lucide-react";
import { EmptyState } from "../ui";
import { ArtifactCard } from "./ArtifactCard";
import type { RunRecord, CheckpointRecord } from "../../types/api";

export interface ArtifactsListProps {
  runs: RunRecord[];
  onLoadCheckpoints?: (id: number) => Promise<CheckpointRecord[]>;
}

export function ArtifactsList({ runs, onLoadCheckpoints }: ArtifactsListProps) {
  const completed = runs.filter((r) => r.status === "completed");

  if (completed.length === 0) {
    return (
      <EmptyState icon={<Boxes size={28} />} title="No completed runs">
        Completed runs and their output directories will appear here.
      </EmptyState>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "var(--space-4)",
        }}
      >
        {completed.map((run) => (
          <ArtifactCard key={run.id} run={run} onLoadCheckpoints={onLoadCheckpoints} />
        ))}
      </div>
      <p style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>
        Pushing adapters and merged models to the Hub is on the roadmap.
      </p>
    </div>
  );
}
