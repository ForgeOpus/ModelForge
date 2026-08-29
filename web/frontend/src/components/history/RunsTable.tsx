import { History } from "lucide-react";
import { Table, THead, TBody, TR, TH, TD, EmptyState } from "../ui";
import { RunStatusBadge } from "./RunStatusBadge";
import { PeftTierBadge } from "../recommendation";
import { relativeTime, formatDateTime } from "../../lib/format";
import type { RunRecord } from "../../types/api";

export interface RunsTableProps {
  runs: RunRecord[];
  onOpen: (run: RunRecord) => void;
}

export function RunsTable({ runs, onOpen }: RunsTableProps) {
  if (runs.length === 0) {
    return (
      <EmptyState icon={<History size={28} />} title="No runs yet">
        Launch a fine-tune from New Run and it will appear here.
      </EmptyState>
    );
  }

  return (
    <Table>
      <THead>
        <TR>
          <TH>ID</TH>
          <TH>Status</TH>
          <TH>Model</TH>
          <TH>Use case</TH>
          <TH>PEFT</TH>
          <TH>Dataset</TH>
          <TH>Created</TH>
        </TR>
      </THead>
      <TBody>
        {runs.map((run) => (
          <TR key={run.id} onClick={() => onOpen(run)}>
            <TD>{run.id}</TD>
            <TD>
              <RunStatusBadge status={run.status} size="sm" />
            </TD>
            <TD>
              <span className="mono truncate" style={{ display: "block", maxWidth: "200px" }}>
                {run.model_id}
              </span>
            </TD>
            <TD>{run.use_case}</TD>
            <TD>
              <PeftTierBadge approach={run.peft ?? ""} size="sm" />
            </TD>
            <TD>
              <span className="truncate" style={{ display: "block", maxWidth: "200px" }}>
                {run.dataset}
              </span>
            </TD>
            <TD title={formatDateTime(run.created_at)}>{relativeTime(run.created_at)}</TD>
          </TR>
        ))}
      </TBody>
    </Table>
  );
}
