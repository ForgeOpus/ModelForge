import { useState } from "react";
import { Card, Tag, Button, Spinner } from "../ui";
import { relativeTime } from "../../lib/format";
import type { RunRecord, CheckpointRecord } from "../../types/api";

export interface ArtifactCardProps {
  run: RunRecord;
  onLoadCheckpoints?: (id: number) => Promise<CheckpointRecord[]>;
}

export function ArtifactCard({ run, onLoadCheckpoints }: ArtifactCardProps) {
  const [copied, setCopied] = useState(false);
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  async function copyDir() {
    try {
      await navigator.clipboard.writeText(run.output_dir);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }

  async function handleToggle(e: React.SyntheticEvent<HTMLDetailsElement>) {
    if (!e.currentTarget.open || loaded || !onLoadCheckpoints) return;
    setLoading(true);
    setLoaded(true);
    try {
      setCheckpoints(await onLoadCheckpoints(run.id));
    } catch {
      setCheckpoints([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
        <span className="mono" style={{ color: "var(--text-1)", fontWeight: 600 }}>
          {run.model_id}
        </span>

        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <span className="mono truncate" style={{ color: "var(--text-2)", flex: 1 }}>
            {run.output_dir}
          </span>
          <Button variant="ghost" size="sm" onClick={copyDir}>
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>

        <div className="cluster">
          <Tag>{run.task_slug ?? run.use_case}</Tag>
          {run.strategy && <Tag>{run.strategy}</Tag>}
          {run.peft && <Tag>{run.peft}</Tag>}
        </div>

        {run.tensorboard_log_dir && (
          <span className="mono" style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>
            modelforge tensorboard {run.tensorboard_log_dir}
          </span>
        )}

        <details onToggle={handleToggle}>
          <summary style={{ cursor: "pointer", color: "var(--text-2)" }}>Checkpoints</summary>
          <div style={{ marginTop: "var(--space-2)" }}>
            {loading ? (
              <Spinner size={16} />
            ) : !checkpoints || checkpoints.length === 0 ? (
              <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>
                No checkpoints recorded.
              </span>
            ) : (
              <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
                {checkpoints.map((cp) => (
                  <li
                    key={cp.id}
                    style={{ display: "flex", gap: "var(--space-2)", fontSize: "var(--step--1)" }}
                  >
                    <span className="mono truncate" style={{ flex: 1 }}>
                      {cp.path}
                    </span>
                    <span style={{ color: "var(--text-3)" }}>step {cp.step ?? "—"}</span>
                    <span style={{ color: "var(--text-3)" }}>{relativeTime(cp.created_at)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </details>
      </div>
    </Card>
  );
}
