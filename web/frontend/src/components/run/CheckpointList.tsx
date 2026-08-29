import { Table, THead, TBody, TR, TH, TD, Button } from "../ui";
import { relativeTime } from "../../lib/format";
import type { CheckpointRecord } from "../../types/api";

export interface CheckpointListProps {
  checkpoints: CheckpointRecord[];
  canResume: boolean;
  onResume: (path: string) => void;
  resuming: boolean;
}

export function CheckpointList({ checkpoints, canResume, onResume, resuming }: CheckpointListProps) {
  if (checkpoints.length === 0) {
    return (
      <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>No checkpoints yet.</span>
    );
  }

  return (
    <Table>
      <THead>
        <TR>
          <TH>Step</TH>
          <TH>Path</TH>
          <TH>Saved</TH>
          {canResume && <TH />}
        </TR>
      </THead>
      <TBody>
        {checkpoints.map((c) => (
          <TR key={c.id}>
            <TD>{c.step ?? "—"}</TD>
            <TD
              className="mono"
              title={c.path}
              style={{
                maxWidth: "320px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {c.path}
            </TD>
            <TD>{relativeTime(c.created_at)}</TD>
            {canResume && (
              <TD>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={resuming}
                  onClick={() => onResume(c.path)}
                >
                  Resume from here
                </Button>
              </TD>
            )}
          </TR>
        ))}
      </TBody>
    </Table>
  );
}
