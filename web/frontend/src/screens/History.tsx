import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { History as HistoryIcon } from "lucide-react";

import { useStore } from "@/lib/store";
import { cn, formatDuration, formatRelative, parseDate } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { EmptyState, Panel, Skeleton } from "@/components/ui/misc";
import { StatusChip, TierChip } from "@/components/TierChip";
import { Link } from "react-router-dom";

/** A run marked failed by mark_interrupted_runs() was never actually running
    when ModelForge stopped — that is not the same as a training failure, and
    colouring it red sends people hunting for a bug that isn't there. */
const INTERRUPTED = "ModelForge stopped before this run";

type Filter = "all" | "completed" | "failed" | "cancelled";

export function History() {
  const { runs, loading } = useStore();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<Filter>("all");

  const shown = useMemo(
    () => (filter === "all" ? runs : runs.filter((r) => r.status === filter)),
    [runs, filter],
  );

  if (loading && runs.length === 0) {
    return (
      <div className="flex flex-col gap-2 p-6">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-11 w-full" />
        ))}
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <EmptyState
        icon={<HistoryIcon />}
        title="Nothing has run yet"
        action={
          <Button variant="primary" asChild>
            <Link to="/app">Configure a fine-tune</Link>
          </Button>
        }
      >
        Every run is recorded here with its plan, its checkpoints, and its outcome.
      </EmptyState>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-4 p-6">
      <div className="flex flex-wrap gap-0.5">
        {(["all", "completed", "failed", "cancelled"] as const).map((f) => (
          <button
            key={f}
            type="button"
            aria-pressed={filter === f}
            onClick={() => setFilter(f)}
            className={cn(
              "rounded-xs border px-2.5 py-1 font-mono text-[11.5px]",
              filter === f
                ? "border-line bg-raised text-text"
                : "border-transparent text-faint hover:text-text",
            )}
          >
            {f}
            {f !== "all" && (
              <span className="ml-1.5 text-faint">
                {runs.filter((r) => r.status === f).length}
              </span>
            )}
          </button>
        ))}
      </div>

      <Panel className="min-w-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-left">
                {["Status", "Run", "Base model", "Goal", "Precision", "Dataset", "Started", "Duration"].map(
                  (h) => (
                    <th
                      key={h}
                      scope="col"
                      className="whitespace-nowrap px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-faint"
                    >
                      {h}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {shown.map((run) => {
                const interrupted = Boolean(run.error?.includes(INTERRUPTED));
                const duration =
                  (parseDate(run.updated_at) - parseDate(run.created_at)) / 1000;
                return (
                  <tr
                    key={run.id}
                    tabIndex={0}
                    onClick={() => navigate(`/app/run/${run.id}`)}
                    onKeyDown={(e) => e.key === "Enter" && navigate(`/app/run/${run.id}`)}
                    className="cursor-pointer border-b border-line last:border-b-0 hover:bg-raised focus-visible:bg-raised"
                  >
                    <td className="whitespace-nowrap px-3 py-2.5">
                      <StatusChip status={interrupted ? "interrupted" : run.status} muted={interrupted} />
                    </td>
                    <td className="px-3 py-2.5 font-mono text-faint">#{run.id}</td>
                    <td className="max-w-[240px] truncate px-3 py-2.5 font-mono">{run.model_id}</td>
                    <td className="whitespace-nowrap px-3 py-2.5 text-dim">
                      {run.use_case}
                      <span className="text-faint"> · {run.objective}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <TierChip peft={run.peft} />
                    </td>
                    <td className="max-w-[180px] truncate px-3 py-2.5 font-mono text-faint">
                      {run.dataset}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2.5 text-dim">
                      {formatRelative(run.created_at)}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2.5 font-mono text-dim tnum">
                      {duration > 0 ? formatDuration(duration) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {runs.some((r) => r.error?.includes(INTERRUPTED)) && (
        <p className="text-[11.5px] leading-relaxed text-faint">
          Runs marked <span className="text-dim">interrupted</span> were still going when ModelForge
          last shut down. Their training did not fail — resume them from their most recent
          checkpoint.
        </p>
      )}
    </div>
  );
}

export { History as HistoryScreen };
