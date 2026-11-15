import pytest
from finetuning_strategies.model_factory import TaskModels, TaskResponse
from finetuning_strategies.tasks.causal_language_modeling import CausalLM


class TestTaskModels:
    def test_get_task_class_causal_lm(self):
        result = TaskModels.get_task_class("causal_language_modeling")
        assert result is CausalLM

    def test_get_task_class_unknown_raises(self):
        with pytest.raises(ValueError, match="seq2seq"):
            TaskModels.get_task_class("seq2seq")

    def test_list_tasks_non_empty(self):
        tasks = TaskModels.list_tasks()
        assert len(tasks) >= 1

    def test_list_tasks_contains_causal_lm(self):
        tasks = TaskModels.list_tasks()
        slugs = [t.slug for t in tasks]
        assert "causal_language_modeling" in slugs

    def test_task_response_model_class_is_string(self):
        tasks = TaskModels.list_tasks()
        for t in tasks:
            assert isinstance(t.model_class, str)

    def test_task_response_fields_complete(self):
        tasks = TaskModels.list_tasks()
        for t in tasks:
            assert isinstance(t, TaskResponse)
            assert t.name
            assert t.slug
            assert t.description
            assert t.model_class
