import { ExternalLink } from "lucide-react";
import { Table, THead, TBody, TR, TH, TD, Button } from "../ui";
import { formatParams, formatInt } from "../../lib/format";
import type { RecommendedModel } from "../../types/api";

export interface ModelCandidateListProps {
  models: RecommendedModel[];
  selectedId?: string;
  onSelect?: (id: string) => void;
}

export function ModelCandidateList({ models, selectedId, onSelect }: ModelCandidateListProps) {
  return (
    <Table>
      <THead>
        <TR>
          <TH>#</TH>
          <TH>Model</TH>
          <TH>Parameters</TH>
          <TH>Downloads</TH>
          <TH />
        </TR>
      </THead>
      <TBody>
        {models.map((model, i) => {
          const selected = selectedId === model.model_id;
          return (
            <TR
              key={model.model_id}
              style={selected ? { background: "var(--accent-soft)" } : undefined}
            >
              <TD>{i + 1}</TD>
              <TD>
                <span className="mono" style={{ overflowWrap: "anywhere" }}>
                  {model.model_id}{" "}
                  <a
                    href={`https://huggingface.co/${model.model_id}`}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: "var(--text-3)", verticalAlign: "middle" }}
                  >
                    <ExternalLink size={12} />
                  </a>
                </span>
              </TD>
              <TD>{formatParams(model.num_parameters)}</TD>
              <TD>{formatInt(model.downloads)}</TD>
              <TD>
                {onSelect && (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={selected}
                    onClick={() => onSelect(model.model_id)}
                  >
                    {selected ? "Selected" : "Select"}
                  </Button>
                )}
              </TD>
            </TR>
          );
        })}
      </TBody>
    </Table>
  );
}
