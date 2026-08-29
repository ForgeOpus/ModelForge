import { ArtifactsList } from "../../components/artifacts";
import { Callout, Spinner } from "../../components/ui";
import { api } from "../../api/client";
import { useRuns } from "../../hooks";

export function ArtifactsPage() {
  const { runs, loading, error } = useRuns({ pollMs: 10000 });

  return (
    <div className="stack" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <h1>Artifacts</h1>
      {error && !runs.length ? (
        <Callout tone="danger" title="Could not load runs">{error.message}</Callout>
      ) : loading && !runs.length ? (
        <div style={{ padding: "var(--space-8)", textAlign: "center" }}><Spinner size={24} /></div>
      ) : (
        <ArtifactsList
          runs={runs}
          onLoadCheckpoints={(id) => api.checkpoints(id).then((r) => r.checkpoints)}
        />
      )}
    </div>
  );
}
