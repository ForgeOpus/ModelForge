import { useState } from "react";
import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import { Eyebrow, Panel } from "@/components/ui/misc";
import { TIERS, TIER_ORDER, bytesLabel } from "@/lib/tiers";

/* The old app had a /technical-details page. The repo now carries a real docs
   tree and a developer guide, so this explains the concepts you need while
   using the interface and points at those for depth, rather than duplicating
   them into a page that will drift. */

const SECTIONS = ["Concepts", "Data formats", "Precision tiers", "Reference"] as const;
type Section = (typeof SECTIONS)[number];

const CONCEPTS = [
  {
    term: "Use case",
    body: "The goal in plain terms — instruction following, summarization, extractive QA, text generation. Picking one resolves which model architecture and trainer are used.",
  },
  {
    term: "Objective",
    body: "How the model learns. Supervised training imitates example answers. Preference training (DPO) learns from pairs where one answer is better than another.",
  },
  {
    term: "Strategy",
    body: "The trainer that runs the job: sft for supervised, dpo for preference. Each one sizes its own batch and memory settings.",
  },
  {
    term: "Precision tier",
    body: "How many bytes each model weight occupies during training. Lower precision fits bigger models on the same card, at some cost in quality.",
  },
  {
    term: "Adapter",
    body: "LoRA trains a small set of extra weights and leaves the base model frozen. What gets written to your output directory is that adapter, not a full copy of the model.",
  },
  {
    term: "Checkpoint",
    body: "A snapshot written every save_steps steps. If a run stops early you can resume from the most recent one instead of starting over.",
  },
];

const FORMATS = [
  {
    name: "Instruction following · supervised",
    columns: "prompt, completion",
    example: '{"prompt": "Summarise this ticket", "completion": "Customer cannot log in…"}',
  },
  {
    name: "Instruction following · preference",
    columns: "prompt, chosen, rejected",
    example: '{"prompt": "Explain LoRA", "chosen": "LoRA trains…", "rejected": "It is a thing."}',
  },
  {
    name: "Text generation",
    columns: "prompt, completion",
    example: '{"prompt": "Once upon a time", "completion": " there was a compiler."}',
  },
  {
    name: "Summarization",
    columns: "document, summary",
    example: '{"document": "The quarterly report…", "summary": "Revenue rose 4%."}',
  },
  {
    name: "Extractive QA",
    columns: "context, question, answers",
    example: '{"context": "…", "question": "When?", "answers": {"text": ["1999"], "answer_start": [42]}}',
  },
];

export function Docs() {
  const [section, setSection] = useState<Section>("Concepts");

  return (
    <div className="flex flex-col gap-5 p-6">
      <div className="flex flex-wrap gap-0.5">
        {SECTIONS.map((s) => (
          <button
            key={s}
            type="button"
            aria-pressed={section === s}
            onClick={() => setSection(s)}
            className={cn(
              "rounded-xs border px-2.5 py-1 text-[12px]",
              section === s
                ? "border-line bg-raised text-text"
                : "border-transparent text-faint hover:text-text",
            )}
          >
            {s}
          </button>
        ))}
      </div>

      {section === "Concepts" && (
        <dl className="grid gap-3 md:grid-cols-2">
          {CONCEPTS.map((c) => (
            <Panel key={c.term} className="p-4">
              <dt className="font-display text-[14px] font-semibold">{c.term}</dt>
              <dd className="mt-1.5 text-[12.5px] leading-relaxed text-dim">{c.body}</dd>
            </Panel>
          ))}
        </dl>
      )}

      {section === "Data formats" && (
        <div className="flex flex-col gap-3">
          <p className="max-w-prose text-[12.5px] leading-relaxed text-dim">
            Datasets load from a local path or a Hugging Face dataset ID. Parquet, Arrow, JSONL,
            JSON, CSV, and TXT are all read directly; point at a folder and every matching file is
            loaded together. Validate on the Configure screen to check your columns before
            committing to a run.
          </p>
          {FORMATS.map((f) => (
            <Panel key={f.name} className="p-4">
              <h3 className="text-[13px] font-medium">{f.name}</h3>
              <p className="mt-1 font-mono text-[11.5px] text-faint">columns: {f.columns}</p>
              <pre className="mt-2.5 overflow-x-auto rounded-sm border border-line bg-surface p-2.5 text-[11.5px] text-dim">
                {f.example}
              </pre>
            </Panel>
          ))}
        </div>
      )}

      {section === "Precision tiers" && (
        <div className="flex flex-col gap-3">
          <p className="max-w-prose text-[12.5px] leading-relaxed text-dim">
            Colour means precision throughout ModelForge. Wherever you see one of these, it is the
            same tier: on a model row, on a run in your history, and as the stroke of a loss curve.
          </p>
          {TIER_ORDER.map((t) => {
            const meta = TIERS[t];
            return (
              <Panel key={t} className="flex items-start gap-3.5 p-4">
                <span
                  className="mt-1 size-3 shrink-0 rounded-xs"
                  style={{ background: meta.cssVar }}
                />
                <div>
                  <h3 className="font-mono text-[13px]" style={{ color: meta.cssVar }}>
                    {t}
                  </h3>
                  <p className="mt-1 text-[12.5px] leading-relaxed text-dim">{meta.blurb}</p>
                  <p className="mt-1 font-mono text-[11.5px] text-faint">
                    {bytesLabel(meta.bytesPerParam)}
                  </p>
                </div>
              </Panel>
            );
          })}
        </div>
      )}

      {section === "Reference" && (
        <div className="flex flex-col gap-3">
          <Panel className="p-4">
            <Eyebrow className="mb-2">In this repository</Eyebrow>
            <ul className="flex flex-col gap-1.5 text-[12.5px] text-dim">
              <li><code className="font-mono text-text">DEVELOPER_GUIDE.md</code> — architecture and how to add a strategy, task, or use case</li>
              <li><code className="font-mono text-text">docs/</code> — installation, configuration, dataset formats, troubleshooting</li>
            </ul>
          </Panel>
          <Panel className="p-4">
            <Eyebrow className="mb-2">Command line</Eyebrow>
            <pre className="overflow-x-auto rounded-sm border border-line bg-surface p-3 text-[11.5px] leading-relaxed text-dim">
{`modelforge run         # interactive fine-tuning wizard
modelforge recommend   # show recommendations without training
modelforge list-tasks  # available use cases
modelforge history     # past training runs
modelforge web         # this interface`}
            </pre>
          </Panel>
          <Panel className="p-4">
            <Eyebrow className="mb-2">Online</Eyebrow>
            <a
              href="https://github.com/ForgeOpus/ModelForge"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-[12.5px] text-dim underline-offset-4 hover:text-text hover:underline"
            >
              github.com/ForgeOpus/ModelForge <ExternalLink className="size-3.5" />
            </a>
          </Panel>
        </div>
      )}
    </div>
  );
}
