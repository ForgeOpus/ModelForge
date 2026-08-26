import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Check, Copy, Github, Moon, Sun } from "lucide-react";

import { HeroMatcher } from "./HeroMatcher";
import { LogoPlate } from "./LogoPlate";
import { motion } from "motion/react";
import { reveal, revealOnScroll, useReducedMotion } from "@/lib/motion";
import { Lockup, Wordmark } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Eyebrow, Panel } from "@/components/ui/misc";
import { useTheme } from "@/lib/theme";
import { TIERS, TIER_ORDER, bytesLabel } from "@/lib/tiers";

/* Brand orange leads here and the graphite supports it -- the reverse of the
   dashboard, where colour has to stay out of the way of the data. */

const STEPS = [
  {
    title: "It reads your hardware",
    body: "ModelForge detects your GPUs, RAM, and disk, then works out how much memory is actually available to train with once activations and the adapter are accounted for.",
  },
  {
    title: "It picks a plan that fits",
    body: "From that budget it resolves a precision tier, queries the Hub for models that fit inside it, and builds a full training configuration — batch size, sequence length, LoRA rank, the lot.",
  },
  {
    title: "You train, and watch it happen",
    body: "Start the run and follow the loss curve, throughput, and checkpoints live. Cancel any time; resume from the last checkpoint when you come back.",
  },
];

export function Landing() {
  const { theme, toggle } = useTheme();
  const reduced = useReducedMotion();
  const [copied, setCopied] = useState(false);

  const install = "pip install modelforge-finetuning";

  async function copy() {
    try {
      await navigator.clipboard.writeText(install);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard permission denied; the command is visible anyway */
    }
  }

  return (
    <div className="min-h-screen bg-ground">
      <header className="border-b border-line">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-4 sm:gap-5 sm:px-6">
          <Wordmark className="text-[15px]" />
          <div className="flex-1" />
          <a
            href="https://github.com/ForgeOpus/ModelForge"
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-1.5 text-[13px] text-dim hover:text-text sm:inline-flex"
          >
            <Github className="size-4" /> GitHub
          </a>
          <button
            type="button"
            onClick={toggle}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            className="rounded-xs p-1.5 text-faint hover:text-text"
          >
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
          <Button variant="primary" size="sm" asChild>
            <Link to="/app">
              <span className="hidden sm:inline">Open ModelForge</span>
              <span className="sm:hidden">Open</span>
              <ArrowRight />
            </Link>
          </Button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 sm:px-6">
        <section className="grid items-center gap-10 py-14 lg:grid-cols-[1.05fr_1fr] lg:py-20">
          <div>
            <LogoPlate className="mb-8 w-[132px] sm:w-[156px]" />
            <motion.h1
              className="font-display text-[clamp(32px,5vw,52px)] font-bold leading-[1.04] tracking-[-0.02em]"
              {...reveal(reduced, { delay: 0.12 })}
            >
              Fine-tune a model on the GPU you already have.
            </motion.h1>
            <motion.p
              className="mt-5 max-w-[52ch] text-[15px] leading-relaxed text-dim"
              {...reveal(reduced, { y: 10, delay: 0.2 })}
            >
              ModelForge reads your hardware, picks a base model and a training recipe that fit
              inside it, and runs the whole job locally. Your data never leaves the machine.
            </motion.p>

            <motion.div
              className="mt-7 flex flex-wrap items-center gap-3"
              {...reveal(reduced, { y: 10, delay: 0.28 })}
            >
              <button
                type="button"
                onClick={copy}
                className="group flex min-w-0 max-w-full items-center gap-3 overflow-x-auto rounded-sm border border-line bg-surface px-3.5 py-2.5 font-mono text-[12px] text-text hover:border-line-strong sm:text-[13px]"
              >
                <span className="text-faint">$</span>
                {install}
                {copied ? (
                  <Check className="size-3.5 text-ok" />
                ) : (
                  <Copy className="size-3.5 text-faint group-hover:text-text" />
                )}
              </button>
              <Button variant="primary" size="lg" asChild>
                <Link to="/app">
                  Open ModelForge <ArrowRight />
                </Link>
              </Button>
            </motion.div>
            <p className="mt-3 text-[12px] text-faint">
              Runs on NVIDIA, Apple Silicon, or CPU. No account, no API key.
            </p>
          </div>

          <motion.div {...reveal(reduced, { y: 14, delay: 0.36, duration: 0.55 })}>
            <HeroMatcher />
          </motion.div>
        </section>

        <motion.section className="border-t border-line py-14" {...revealOnScroll(reduced)}>
          <Eyebrow className="mb-8">How it works</Eyebrow>
          <ol className="grid gap-6 md:grid-cols-3">
            {STEPS.map((step, i) => (
              <li key={step.title}>
                {/* Numbered because this genuinely is a sequence someone is
                    being walked through for the first time. */}
                <span className="font-mono text-[12px] text-brand">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <h3 className="mt-2 font-display text-[17px] font-semibold leading-snug">
                  {step.title}
                </h3>
                <p className="mt-2 text-[13px] leading-relaxed text-dim">{step.body}</p>
              </li>
            ))}
          </ol>
        </motion.section>

        <motion.section className="grid items-start gap-10 border-t border-line py-14 lg:grid-cols-[1fr_1.1fr]" {...revealOnScroll(reduced)}>
          <div>
            <Eyebrow className="mb-4">It shows its working</Eyebrow>
            <h2 className="font-display text-[26px] font-bold leading-tight tracking-[-0.01em]">
              Every choice comes with a reason.
            </h2>
            <p className="mt-4 max-w-[46ch] text-[14px] leading-relaxed text-dim">
              Most tools hand you a config and leave you guessing. ModelForge tells you the memory
              it measured, how your goal resolved to a trainer, why it picked one precision over
              another, and what it excluded. If you disagree, every value is editable before you
              start.
            </p>
          </div>

          <Panel className="bg-surface p-5">
            <Eyebrow className="mb-4">Why this plan</Eyebrow>
            <div className="flex flex-col gap-3.5">
              <Reason
                label="Hardware"
                text="24.0 GB across 1 GPU. Adjusted VRAM (after strategy overhead): 24.00 GB."
              />
              <Reason
                label="Routing"
                text="instruction_following + supervised → causal_language_modeling / sft"
              />
              <Reason
                label="Precision"
                text="Selected tier: lora_bf16 (P_max=10.95B). Per-tier P_max: full_finetune 1.22B, lora_bf16 10.95B, qlora_8bit 21.90B, qlora_4bit 43.80B"
              />
              <Reason
                label="Candidates"
                text="Queried Hub: pipeline_tag='text-generation', sort=downloads desc, max_params=10.95B. Found 5 model(s)."
              />
            </div>
          </Panel>
        </motion.section>

        <motion.section className="border-t border-line py-14" {...revealOnScroll(reduced)}>
          <Eyebrow className="mb-8">Precision, and what it costs you</Eyebrow>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {TIER_ORDER.map((t) => {
              const meta = TIERS[t];
              return (
                <Panel key={t} className="p-4">
                  <span
                    className="mb-2.5 block h-1 w-9 rounded-full"
                    style={{ background: meta.cssVar }}
                  />
                  <h3 className="font-mono text-[13px]" style={{ color: meta.cssVar }}>
                    {t}
                  </h3>
                  <p className="mt-2 text-[12.5px] leading-relaxed text-dim">{meta.blurb}</p>
                  <p className="mt-2.5 font-mono text-[11px] text-faint">
                    {bytesLabel(meta.bytesPerParam)}
                  </p>
                </Panel>
              );
            })}
          </div>
          <p className="mt-5 max-w-[64ch] text-[12.5px] leading-relaxed text-faint">
            These four colours mean the same thing everywhere in ModelForge — on a model you are
            choosing, on a run in your history, and as the stroke of a loss curve. You can see what
            a run cost without reading a single number.
          </p>
        </motion.section>

        <motion.section className="border-t border-line py-14" {...revealOnScroll(reduced)}>
          <Eyebrow className="mb-8">What it does today</Eyebrow>
          <div className="grid gap-8 md:grid-cols-3">
            <Capability
              title="Trains"
              items={[
                "Instruction following",
                "Text generation",
                "Summarization",
                "Extractive QA",
                "Supervised (SFT) and preference (DPO)",
              ]}
            />
            <Capability
              title="Handles"
              items={[
                "Parquet, Arrow, JSONL, JSON, CSV, TXT",
                "Local folders and Hugging Face datasets",
                "Multi-GPU (VRAM summed)",
                "Checkpoints and resume",
                "Live loss, throughput, TensorBoard",
              ]}
            />
            <Capability
              title="Then"
              items={[
                "Chat with the result in the playground",
                "Push to the Hugging Face Hub",
                "Keep every run in one history",
              ]}
              footnote="Adapters are saved unmerged. GGUF and Ollama export are not built yet."
            />
          </div>
        </motion.section>

        <motion.section className="border-t border-line py-16 text-center" {...revealOnScroll(reduced, { amount: 0.2 })}>
          <h2 className="font-display text-[28px] font-bold tracking-[-0.01em]">
            Point it at your data and see what your card can do.
          </h2>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            <Button variant="primary" size="lg" asChild>
              <Link to="/app">
                Open ModelForge <ArrowRight />
              </Link>
            </Button>
            <Button variant="secondary" size="lg" asChild>
              <a href="https://github.com/ForgeOpus/ModelForge" target="_blank" rel="noreferrer">
                <Github /> Read the source
              </a>
            </Button>
          </div>
        </motion.section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-6 text-[12px] text-faint sm:px-6">
          <Lockup size="sm" />
          <span className="flex-1" />
          <span>Runs entirely on your machine.</span>
        </div>
      </footer>
    </div>
  );
}

function Reason({ label, text }: { label: string; text: string }) {
  return (
    <article>
      <h4 className="eyebrow mb-1 text-[10px]">{label}</h4>
      <p className="m-0 whitespace-pre-wrap text-[12px] leading-relaxed text-dim">{text}</p>
    </article>
  );
}

function Capability({
  title,
  items,
  footnote,
}: {
  title: string;
  items: string[];
  footnote?: string;
}) {
  return (
    <div>
      <h3 className="font-display text-[15px] font-semibold">{title}</h3>
      <ul className="mt-3 flex flex-col gap-2">
        {items.map((item) => (
          <li key={item} className="flex gap-2 text-[13px] leading-snug text-dim">
            <Check className="mt-0.5 size-3.5 shrink-0 text-brand" />
            {item}
          </li>
        ))}
      </ul>
      {footnote && <p className="mt-3 text-[11.5px] leading-relaxed text-faint">{footnote}</p>}
    </div>
  );
}
