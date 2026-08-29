import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { RunStatusBadge, RunSummaryCard, CheckpointList } from "../../components/run";
import { RationaleReport } from "../../components/recommendation";
import { Callout, Card, Spinner } from "../../components/ui";
import { api } from "../../api/client";
import { useRun } from "../../hooks";
import { parseRunConfig, type CheckpointRecord } from "../../types/api";
import { statusMeta, formatDateTime } from "../../lib/format";

export function RunDetailPage() {
  const { runId } = useParams();
  const id = runId ? Number(runId) : null;
  const { data: run, error, loading } = useRun(id);
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);

  const loadCheckpoints = useCallback(async () => {
    if (id == null) return;
    try {
      const { checkpoints: cps } = await api.checkpoints(id);
      setCheckpoints(cps);
    } catch {
      /* ignore */
    }
  }, [id]);

  useEffect(() => {
    void loadCheckpoints();
  }, [loadCheckpoints]);

  if (loading && !run) {
    return <div style={{ padding: "var(--space-8)", textAlign: "center" }}><Spinner size={28} /></div>;
  }
  if (error && !run) {
    return <Callout tone="danger" title={`Run #${runId}`}>{error.message}</Callout>;
  }
  if (!run) return null;

  const rec = parseRunConfig(run.config_json);
  const active = statusMeta(run.status).active;

  return (
    <div className="stack" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <div className="cluster" style={{ justifyContent: "space-between" }}>
        <div className="cluster">
          <h1 style={{ margin: 0 }}>Run #{run.id}</h1>
          <RunStatusBadge status={run.status} />
        </div>
        {active && <Link to={`/app/runs/${run.id}`}>Open live view →</Link>}
      </div>

      <RunSummaryCard run={run} />

      <Card>
        <p className="muted" style={{ margin: 0 }}>
          Created {formatDateTime(run.created_at)} · updated {formatDateTime(run.updated_at)}
        </p>
      </Card>

      <CheckpointList checkpoints={checkpoints} canResume={false} onResume={() => {}} resuming={false} />

      {rec ? (
        <div>
          <h2>Launch recommendation</h2>
          <RationaleReport recommendation={rec} />
        </div>
      ) : (
        <Callout tone="info" title="No recommendation stored">
          This run has no saved recommendation payload.
        </Callout>
      )}
    </div>
  );
}
