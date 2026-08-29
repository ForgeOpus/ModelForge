import { Badge } from "../ui";
import { peftMeta } from "../../lib/format";

export interface PeftTierBadgeProps {
  approach: string;
  size?: "sm" | "md";
}

export function PeftTierBadge({ approach }: PeftTierBadgeProps) {
  const meta = peftMeta(approach);
  return <Badge color={meta.color}>{meta.label}</Badge>;
}
