import { Select, Input } from "../ui";

export interface RunFiltersProps {
  status: string;
  query: string;
  onStatusChange: (s: string) => void;
  onQueryChange: (q: string) => void;
}

const STATUS_OPTIONS = ["all", "queued", "running", "completed", "failed", "cancelled"];

export function RunFilters({ status, query, onStatusChange, onQueryChange }: RunFiltersProps) {
  return (
    <div className="cluster">
      <Select value={status} onChange={(e) => onStatusChange(e.target.value)}>
        {STATUS_OPTIONS.map((opt) => (
          <option key={opt} value={opt}>
            {opt === "all" ? "All statuses" : opt[0].toUpperCase() + opt.slice(1)}
          </option>
        ))}
      </Select>
      <Input
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder="Filter by model, dataset or output dir"
      />
    </div>
  );
}
