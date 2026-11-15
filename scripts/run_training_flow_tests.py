#!/usr/bin/env python3
"""Exercise every documented fine-tuning flow with tiny local models.

This is an integration harness, not a pytest wrapper. Each flow creates a
real Hugging Face-compatible model and dataset locally, calls the shared
``workflow.training.assemble_and_train`` boundary, and verifies a trained
model artifact. No Hub download, browser, GPU, or run-monitor is required.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "test-results" / "training-flows.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Flow:
    name: str
    use_case: str
    objective: str
    model_kind: str
    rows: dict[str, list]


# The complete user-facing data contracts from TRAINING_WORKFLOWS.md.
FLOWS = (
    Flow("instruction-following-supervised", "instruction_following", "supervised", "causal", {"instruction": ["say hello"], "response": ["hello"]}),
    Flow("text-generation-supervised", "text_generation", "supervised", "causal", {"input_text": ["once"], "generated_text": [" hello"]}),
    Flow("instruction-following-preference", "instruction_following", "preference", "causal", {"instruction": ["say hello"], "preferred_response": ["hello"], "dispreferred_response": ["once"]}),
    Flow("text-generation-preference", "text_generation", "preference", "causal", {"input_text": ["once"], "preferred_continuation": [" hello"], "rejected_continuation": [" say"]}),
    Flow("instruction-following-reinforcement", "instruction_following", "reinforcement", "causal", {"instruction": ["say hello", "say once"], "ideal_answer": ["hello", "once"]}),
    Flow("text-generation-reinforcement", "text_generation", "reinforcement", "causal", {"input_text": ["say", "once"], "ideal_continuation": [" hello", " say"]}),
    Flow("summarization-supervised", "summarization", "supervised", "seq2seq", {"document": ["say hello"], "summary": ["hello"]}),
    Flow("extractive-qa-supervised", "extractive_qa", "supervised", "qa", {"context": ["paris is in france ."], "question": ["where is paris"], "answers": [{"text": ["france"], "answer_start": [12]}]}),
)


def build_local_models(root: Path) -> dict[str, Path]:
    """Create only the tiny base models needed for real CPU training."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertConfig, BertForQuestionAnswering, BertTokenizerFast, GPT2Config,
        GPT2LMHeadModel, PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration,
    )
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "say": 3, "hello": 4, "once": 5, "paris": 6, "is": 7, "in": 8, "france": 9, ".": 10, "where": 11}
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    word_level = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    word_level.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=word_level, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")

    causal_dir = models / "causal"
    fast.save_pretrained(causal_dir)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=64, n_embd=8, n_layer=1, n_head=1, eos_token_id=2, pad_token_id=1)).save_pretrained(causal_dir)

    seq2seq_dir = models / "seq2seq"
    fast.save_pretrained(seq2seq_dir)
    T5ForConditionalGeneration(T5Config(vocab_size=len(vocabulary), d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=1, decoder_start_token_id=1, pad_token_id=1, eos_token_id=2)).save_pretrained(seq2seq_dir)

    qa_dir = models / "qa"
    qa_dir.mkdir()
    bert_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "where", "is", "paris", "in", "france", "."]
    (qa_dir / "vocab.txt").write_text("\n".join(bert_vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(qa_dir / "vocab.txt")).save_pretrained(qa_dir)
    BertForQuestionAnswering(BertConfig(vocab_size=len(bert_vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=1, intermediate_size=16)).save_pretrained(qa_dir)
    return {"causal": causal_dir, "seq2seq": seq2seq_dir, "qa": qa_dir}


def strategy_config(strategy_slug: str) -> dict[str, object]:
    common: dict[str, object] = {"bf16": False, "fp16": False, "num_train_epochs": 1, "per_device_train_batch_size": 1, "gradient_checkpointing": False, "report_to": "none", "logging_steps": 1000, "save_steps": 1000}
    if strategy_slug == "sft":
        return {**common, "max_length": 16}
    if strategy_slug == "dpo":
        return {**common, "max_length": 16, "gradient_accumulation_steps": 1, "use_liger_kernel": False}
    if strategy_slug == "grpo":
        return {**common, "max_completion_length": 2, "num_generations": 2, "gradient_accumulation_steps": 2, "beta": 0.0, "use_liger_kernel": False}
    if strategy_slug == "qa":
        return {**common, "max_length": 32, "doc_stride": 8}
    raise ValueError(f"No integration configuration for strategy '{strategy_slug}'.")


def recommendation_for(flow: Flow):
    from recommendation_engine.validation_models.recommendation import Rationale, Recommendation, RecommendedModel
    from utilities.parallelism_strategy import ParallelismStrategy
    from utilities.recommendation_engine import RecommendationEngine
    from utilities.training_objective import TrainingObjective

    _, _, task_slug, strategy_slug, data_format = RecommendationEngine._resolve_use_case_and_strategy(
        flow.use_case, TrainingObjective(flow.objective)
    )
    return Recommendation(
        recommended_models=[RecommendedModel(model_id="local/tiny", num_parameters=1, downloads=0)],
        task_slug=task_slug, strategy_slug=strategy_slug, peft_approach="full_finetune", provider_slug="huggingface",
        strategy_config=strategy_config(strategy_slug), peft_config=None, data_format=data_format,
        parallelism_strategy=ParallelismStrategy.DATA_PARALLEL,
        rationale=Rationale(vram_summary="local CPU integration harness", use_case_resolution="resolved locally", peft_decision="full fine-tune for a tiny test model", model_filter="no Hub lookup"),
    )


def run_flow(flow: Flow, model_paths: dict[str, Path], artifacts_root: Path) -> dict[str, object]:
    """Train one documented flow and capture its exception without stopping peers."""
    started = monotonic()
    output_dir = artifacts_root / flow.name
    try:
        from datasets import Dataset
        from workflow.training import assemble_and_train

        source = Dataset.from_dict(flow.rows)
        assemble_and_train(recommendation_for(flow), str(model_paths[flow.model_kind]), source, str(output_dir), "", callbacks=[])
        config = output_dir / "config.json"
        if not config.is_file():
            raise AssertionError(f"Training returned without creating {config}.")
        return {"flow": flow.name, "status": "passed", "duration_seconds": round(monotonic() - started, 3), "source_columns": source.column_names, "trained_model_artifact": "config.json"}
    except Exception as error:
        return {"flow": flow.name, "status": "failed", "duration_seconds": round(monotonic() - started, 3), "source_columns": list(flow.rows), "error": str(error), "traceback": traceback.format_exc()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report destination.")
    parser.add_argument("--flow", action="append", choices=[flow.name for flow in FLOWS], help="Run only this flow; repeatable.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep generated tiny models and trained outputs.")
    parser.add_argument("--list", action="store_true", help="List flows and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print("\n".join(flow.name for flow in FLOWS))
        return 0
    selected = [flow for flow in FLOWS if not args.flow or flow.name in args.flow]
    temporary_root = None
    if args.keep_artifacts:
        artifacts_root = ROOT / "test-results" / "training-flow-artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
    else:
        temporary_root = tempfile.TemporaryDirectory(prefix="modelforge-training-flows-")
        artifacts_root = Path(temporary_root.name)
    try:
        model_paths = build_local_models(artifacts_root)
        results = [run_flow(flow, model_paths, artifacts_root / "outputs") for flow in selected]
    except Exception as error:
        results = [{"flow": "harness-setup", "status": "failed", "error": str(error), "traceback": traceback.format_exc()}]
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()
    failures = [result["flow"] for result in results if result["status"] != "passed"]
    report = {"started_at": datetime.now(UTC).isoformat(), "repository": str(ROOT), "kind": "local end-to-end fine-tuning integration test", "status": "passed" if not failures else "failed", "failed_flows": failures, "results": results}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Training-flow results exported to {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
