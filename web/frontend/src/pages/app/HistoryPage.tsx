import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { RunFilters, RunsTable } from "../../components/history";
import { Callout, Spinner } from "../../components/ui";
import { useRuns } from "../../hooks";
import type { RunRecord } from "../../types/api";

export function HistoryPage() {
  const navigate = useNavigate();
  const { runs, loading, error } = useRuns({ pollMs: 5000 });
  const [status, setStatus] = useState("all");
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return runs.filter((r) => {
      if (status !== "all" && r.status !== status) return false;
      if (!q) return true;
      return (
        r.model_id.toLowerCase().includes(q) ||
        r.dataset.toLowerCase().includes(q) ||
        r.output_dir.toLowerCase().includes(q)
      );
    });
  }, [runs, status, query]);

  return (
    <div className="stack" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <h1>Run history</h1>
      <RunFilters
        status={status}
        query={query}
        onStatusChange={setStatus}
        onQueryChange={setQuery}
      />
      {error && !runs.length ? (
        <Callout tone="danger" title="Could not load runs">{error.message}</Callout>
      ) : loading && !runs.length ? (
        <div style={{ padding: "var(--space-8)", textAlign: "center" }}><Spinner size={24} /></div>
      ) : (
        <RunsTable runs={filtered} onOpen={(run: RunRecord) => navigate(`/app/history/${run.id}`)} />
      )}
    </div>
  );
}
