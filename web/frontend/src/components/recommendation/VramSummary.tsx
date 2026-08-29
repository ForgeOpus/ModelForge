import { Cpu } from "lucide-react";
import { Callout } from "../ui";
import { splitGbEmphasis } from "../../lib/format";

export interface VramSummaryProps {
  text: string;
}

export function VramSummary({ text }: VramSummaryProps) {
  const parts = splitGbEmphasis(text);
  return (
    <Callout tone="info" title="Hardware" icon={<Cpu size={18} />}>
      <span style={{ color: "var(--text-2)", lineHeight: "var(--leading-normal)" }}>
        {parts.map((part, i) =>
          part.emphasis ? (
            <strong key={i} style={{ color: "var(--text-1)" }}>
              {part.text}
            </strong>
          ) : (
            <span key={i}>{part.text}</span>
          ),
        )}
      </span>
    </Callout>
  );
}
