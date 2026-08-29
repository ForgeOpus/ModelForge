import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, Link } from "react-router-dom";
import { Activity } from "lucide-react";
import {
  RunConsole,
  RunProgress,
  RunStatusBadge,
  RunSummaryCard,
  CheckpointList,
} from "../../components/run";
import { RationaleReport } from "../../components/recommendation";
import { Button, Callout, Card, Dialog, EmptyState, Spinner } from "../../components/ui";
import { api, ApiError } from "../../api/client";
import { useRun, useRunEvents } from "../../hooks";
import { useRunActivity } from "../../context/RunActivityContext";
import { useToast } from "../../components/ui/Toast";
import { isTerminal, type TerminalEvent } from "../../types/events";
import { parseRunConfig, type CheckpointRecord } from "../../types/api";
import { statusMeta } from "../../lib/format";
import "./pages.css";

export function LiveRunPage() {
  const params = useParams();
  const navigate = useNavigate();
  const { push } = useToast();
  const { activeRunId, setActiveRunId, refreshActivity } = useRunActivity();

  const routeId = params.runId ? Number(params.runId) : null;
  const runId = routeId ?? activeRunId;

  const { data: run, error, loading, reload } = useRun(runId);
  const runActive = run ? statusMeta(run.status).active : false;
  const { events, status: wsStatus, lastLog } = useRunEvents(runActive ? runId : null);

  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [resuming, setResuming] = useState(false);
  const seenTerminal = useRef(false);

  const loadCheckpoints = useCallback(async () => {
    if (runId == null) return;
    try {
      const { checkpoints: cps } = await api.checkpoints(runId);
      setCheckpoints(cps);
    } catch {
      /* ignore */
    }
  }, [runId]);

  useEffect(() => {
    seenTerminal.current = false;
    void loadCheckpoints();
  }, [runId, loadCheckpoints]);

  // React to a terminal event on the live stream.
  useEffect(() => {
    const last = events[events.length - 1];
    if (!last || seenTerminal.current) return;
    if (isTerminal(last.type)) {
      seenTerminal.current = true;
      setActiveRunId(null);
      void refreshActivity();
      void loadCheckpoints();
      reload();
    } else if (last.type === "checkpoint") {
      void loadCheckpoints();
    }
  }, [events, setActiveRunId, refreshActivity, loadCheckpoints, reload]);

  async function doCancel() {
    if (runId == null) return;
    setCancelling(true);
    try {
      await api.cancelRun(runId);
      setConfirmCancel(false);
      reload();
    } catch (err) {
      push({ title: "Could not cancel", description: err instanceof Error ? err.message : String(err), tone: "danger" });
    } finally {
      setCancelling(false);
    }
  }

  async function doResume(checkpointPath: string) {
    if (runId == null) return;
    setResuming(true);
    try {
      const resumed = await api.resumeRun(runId, checkpointPath);
      setActiveRunId(resumed.id);
      void refreshActivity();
      navigate(`/app/runs/${resumed.id}`);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        push({ title: "A run is already active", description: "Cancel it before resuming from a checkpoint.", tone: "danger" });
      } else {
        push({ title: "Could not resume", description: err instanceof Error ? err.message : String(err), tone: "danger" });
      }
    } finally {
      setResuming(false);
    }
  }

  if (runId == null) {
    return (
      <EmptyState
        icon={<Activity size={22} />}
        title="No active run"
        children={
          <>
            Nothing is training right now. Start a fine-tune from{" "}
            <Link to="/app">New Run</Link>, or open a past run in{" "}
            <Link to="/app/history">History</Link>.
          </>
        }
      />
    );
  }

  if (loading && !run) {
    return <div style={{ padding: "var(--space-8)", textAlign: "center" }}><Spinner size={28} /></div>;
  }

  if (error && !run) {
    return (
      <Callout tone="danger" title={`Run #${runId}`}>
        {error.message}
      </Callout>
    );
  }

  if (!run) return null;

  const storedRec = parseRunConfig(run.config_json);
  const terminalEvent = events.find(
    (e): e is TerminalEvent => e.type !== "heartbeat" && isTerminal(e.type),
  );

  return (
    <div className="stack" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <div className="cluster" style={{ justifyContent: "space-between" }}>
        <div className="cluster">
          <h1 style={{ margin: 0 }}>Run #{run.id}</h1>
          <RunStatusBadge status={run.status} />
        </div>
        {runActive && (
          <Button variant="danger" onClick={() => setConfirmCancel(true)}>
            Cancel run
          </Button>
        )}
      </div>

      <RunSummaryCard run={run} />

      {runActive && <RunProgress lastLog={lastLog} status={run.status} />}

      {terminalEvent && (
        <Callout
          tone={terminalEvent.type === "completed" ? "success" : terminalEvent.type === "cancelled" ? "warning" : "danger"}
          title={`Run ${terminalEvent.type}`}
        >
          {terminalEvent.type === "completed" ? (
            <>
              Output written to <code className="mono">{terminalEvent.payload.output_dir ?? run.output_dir}</code>.{" "}
              <Link to="/app/artifacts">View in Artifacts</Link>.
            </>
          ) : (
            <span style={{ whiteSpace: "pre-wrap" }}>
              {terminalEvent.payload.error ?? terminalEvent.message}
            </span>
          )}
        </Callout>
      )}

      {runActive ? (
        <RunConsole events={events} connectionStatus={wsStatus} />
      ) : (
        <Card>
          <p className="muted" style={{ margin: 0 }}>
            This run has finished — the live event stream is only available while training is in
            progress. The recommendation it launched with and any error are shown below.
          </p>
        </Card>
      )}

      <CheckpointList
        checkpoints={checkpoints}
        canResume={!runActive && run.status !== "completed"}
        onResume={doResume}
        resuming={resuming}
      />

      {storedRec && (
        <div>
          <h2>Launch recommendation</h2>
          <RationaleReport recommendation={storedRec} compact />
        </div>
      )}

      <Dialog
        open={confirmCancel}
        onClose={() => setConfirmCancel(false)}
        title={`Cancel run #${run.id}?`}
        footer={
          <div className="cluster" style={{ justifyContent: "flex-end" }}>
            <Button variant="ghost" onClick={() => setConfirmCancel(false)}>Keep training</Button>
            <Button variant="danger" loading={cancelling} onClick={doCancel}>Cancel run</Button>
          </div>
        }
      >
        Training stops after the current step. Checkpoints already written are kept and you can
        resume from them.
      </Dialog>
    </div>
  );
}
