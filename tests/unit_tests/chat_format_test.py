from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset

from utilities.chat_format import chat_inputs, normalize_preference, normalize_supervised, map_evaluation


def test_text_preferences_and_supervised_pairs_share_chat_contract():
    pair = normalize_preference({"prompt": "Question?", "chosen": "Yes.", "rejected": "No."})
    supervised = normalize_supervised({"prompt": "Question?", "completion": "Yes."})
    assert pair["prompt"] == supervised["prompt"] == [{"role": "user", "content": "Question?"}]
    assert pair["chosen"] == supervised["completion"] == [{"role": "assistant", "content": "Yes."}]


@pytest.mark.parametrize("prompt", [None, "Question?", [{"role": "user", "content": "Question?"}]])
def test_full_preference_conversations_do_not_duplicate_prompt(prompt):
    row = {
        "chosen": [{"role": "user", "content": "Question?"}, {"role": "assistant", "content": "Yes."}],
        "rejected": [{"role": "user", "content": "Question?"}, {"role": "assistant", "content": "No."}],
    }
    if prompt is not None:
        row["prompt"] = prompt
    normalized = normalize_preference(row)
    assert len(normalized["prompt"]) == len(normalized["chosen"]) == len(normalized["rejected"]) == 1
    assert normalize_preference(normalized) == normalized


def test_assistant_only_pairs_retain_text_prompt_and_evaluation_is_normalized():
    dataset = Dataset.from_list([{
        "prompt": "Question?", "chosen": [{"role": "assistant", "content": "Yes."}],
        "rejected": [{"role": "assistant", "content": "No."}],
    }])
    result = map_evaluation({"validation": dataset}, normalize_preference)
    assert result["validation"][0]["prompt"] == [{"role": "user", "content": "Question?"}]


def test_conflicting_conversation_prompt_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        normalize_preference({
            "prompt": "Different question", "chosen": [{"role": "user", "content": "Question?"}, {"role": "assistant", "content": "Yes."}],
            "rejected": [{"role": "user", "content": "Question?"}, {"role": "assistant", "content": "No."}],
        })


def test_native_template_is_tokenized_once_without_duplicate_bos():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3}, unk_token="[UNK]"))
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.chat_template = "{{ bos_token }}hello"
    inputs = chat_inputs(tokenizer, [{"role": "user", "content": "hello"}])
    assert inputs["input_ids"][0].tolist() == [1, 3]


def test_tensorboard_callback_writes_scalars_to_configured_run_directory(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from transformers import TrainingArguments, TrainerState, TrainerControl
    from transformers.integrations import TensorBoardCallback
    from finetuning_strategies.strategies.sft import SFT

    monkeypatch.setenv("TENSORBOARD_LOGGING_DIR", str(tmp_path / "wrong"))
    log_dir = tmp_path / "run-logs"
    args = TrainingArguments(output_dir=str(tmp_path), logging_dir=str(log_dir), report_to=["tensorboard"], bf16=False)
    callback = TensorBoardCallback()
    trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=[callback]))
    with patch.object(SFT, "_create_tensorboard_program"), patch.object(SFT, "_is_main_process", return_value=True):
        SFT().launch_tensorboard(args, trainer=trainer, open_browser=False)
    state = TrainerState(is_world_process_zero=True, global_step=1)
    control = TrainerControl()
    callback.on_train_begin(args, state, control)
    callback.on_log(args, state, control, logs={"loss": 0.5})
    callback.on_train_end(args, state, control)
    events = EventAccumulator(str(log_dir)).Reload()
    assert events.Scalars("train/loss")[0].value == 0.5
    assert not (tmp_path / "wrong").exists()


def test_isolated_runs_write_distinct_curves_and_reuse_parent_viewer(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from transformers import TrainingArguments, TrainerState, TrainerControl
    from transformers.integrations import TensorBoardCallback
    from finetuning_strategies.strategies.sft import SFT
    from utilities.tensorboard import isolated_log_dir

    with patch.object(SFT, '_create_tensorboard_program') as program, patch.object(SFT, '_is_main_process', return_value=True):
        program.return_value.launch.return_value = 'http://127.0.0.1:6006/'
        for run_id, loss in [(10, 0.4), (11, 0.8)]:
            directory = isolated_log_dir(str(tmp_path), run_id)
            args = TrainingArguments(output_dir=str(tmp_path), logging_dir=directory, report_to=['tensorboard'], bf16=False)
            callback = TensorBoardCallback()
            trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=[callback]))
            SFT().launch_tensorboard(args, trainer=trainer, open_browser=False)
            state, control = TrainerState(is_world_process_zero=True, global_step=1), TrainerControl()
            callback.on_train_begin(args, state, control)
            callback.on_log(args, state, control, logs={'loss': loss})
            callback.on_train_end(args, state, control)
            values = EventAccumulator(directory).Reload().Scalars('train/loss')
            assert len(values) == 1
            assert values[0].value == pytest.approx(loss)
        program.assert_called_once()
        argv = program.return_value.configure.call_args.kwargs['argv']
        assert argv[argv.index('--logdir') + 1] == str(tmp_path)
