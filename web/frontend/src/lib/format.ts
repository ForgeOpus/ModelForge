import type { RunStatus } from "../types/api";

/** 7_000_000_000 -> "7.00B", 340_000_000 -> "340.00M". */
export function formatParams(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return String(n);
}

/** Thousands separators. */
export function formatInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US");
}

export function formatGb(n: number | string | null | undefined, digits = 1): string {
  if (n === "N/A") return "N/A";
  const v = typeof n === "string" ? Number(n) : n;
  if (v == null || !Number.isFinite(v)) return "—";
  return `${v.toFixed(digits)} GB`;
}

/**
 * The backend stores timestamps as `datetime.utcnow().isoformat()` — ISO 8601
 * with no timezone suffix. JS `new Date()` reads those as *local* time, so we
 * append `Z` to mark them UTC before parsing.
 */
function toDate(input: string | number | Date): Date {
  if (typeof input === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(input) && !/[Z+]|-\d{2}:\d{2}$/.test(input.slice(10))) {
    return new Date(`${input}Z`);
  }
  return new Date(input);
}

const RELATIVE = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["year", 31536000],
  ["month", 2592000],
  ["week", 604800],
  ["day", 86400],
  ["hour", 3600],
  ["minute", 60],
  ["second", 1],
];

export function relativeTime(input: string | number | Date | null | undefined): string {
  if (input == null) return "—";
  const then = toDate(input).getTime();
  if (Number.isNaN(then)) return "—";
  const diff = (then - Date.now()) / 1000;
  const abs = Math.abs(diff);
  for (const [unit, secs] of UNITS) {
    if (abs >= secs || unit === "second") {
      return RELATIVE.format(Math.round(diff / secs), unit);
    }
  }
  return "—";
}

export function formatDateTime(input: string | number | Date | null | undefined): string {
  if (input == null) return "—";
  const d = toDate(input);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export interface StatusMeta {
  label: string;
  /** CSS color token reference, e.g. "var(--success)". */
  color: string;
  /** true while the run can still produce live events. */
  active: boolean;
}

export function statusMeta(status: RunStatus | string | null | undefined): StatusMeta {
  switch (status) {
    case "running":
      return { label: "Running", color: "var(--accent)", active: true };
    case "queued":
      return { label: "Queued", color: "var(--info)", active: true };
    case "completed":
      return { label: "Completed", color: "var(--success)", active: false };
    case "failed":
      return { label: "Failed", color: "var(--danger)", active: false };
    case "cancelled":
      return { label: "Cancelled", color: "var(--warning)", active: false };
    default:
      return { label: status ? String(status) : "Unknown", color: "var(--text-3)", active: false };
  }
}

export type PeftApproach =
  | "full_finetune"
  | "lora_bf16"
  | "qlora_8bit"
  | "qlora_4bit"
  | string;

export function peftMeta(approach: PeftApproach | null | undefined): {
  label: string;
  color: string;
} {
  switch (approach) {
    case "full_finetune":
      return { label: "Full fine-tune", color: "var(--peft-full)" };
    case "lora_bf16":
      return { label: "LoRA (bf16)", color: "var(--peft-lora)" };
    case "qlora_8bit":
      return { label: "QLoRA (8-bit)", color: "var(--peft-q8)" };
    case "qlora_4bit":
      return { label: "QLoRA (4-bit)", color: "var(--peft-q4)" };
    default:
      return { label: approach ? String(approach) : "—", color: "var(--text-3)" };
  }
}

export function parallelismLabel(p: string | null | undefined): string {
  if (p === "fsdp") return "FSDP";
  if (p === "data_parallel") return "Data parallel";
  return p ? String(p) : "—";
}

/** Emphasise "<n> GB" spans inside a sentence by wrapping them. */
export function splitGbEmphasis(text: string): Array<{ text: string; emphasis: boolean }> {
  const parts: Array<{ text: string; emphasis: boolean }> = [];
  const re = /(\d+(?:\.\d+)?\s?GB)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index), emphasis: false });
    parts.push({ text: m[0], emphasis: true });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last), emphasis: false });
  return parts;
}
