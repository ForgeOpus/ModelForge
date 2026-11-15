import pytest
from transformers import AutoModelForCausalLM, AutoModel
from finetuning_strategies.tasks.task import FinetuningTask
from finetuning_strategies.tasks.causal_language_modeling import CausalLM


@pytest.fixture(autouse=True)
def clean_task_registry():
    original = dict(FinetuningTask._registry)
    yield
    FinetuningTask._registry.clear()
    FinetuningTask._registry.update(original)


class TestFinetuningTaskRegistration:
    def test_valid_subclass_auto_registers(self):
        class TestTask(FinetuningTask):
            name = "Test Task"
            slug = "test_task_unique_1"
            description = "A test task"
            model_class = AutoModelForCausalLM

        assert "test_task_unique_1" in FinetuningTask._registry
        assert FinetuningTask._registry["test_task_unique_1"] is TestTask

    def test_slug_none_raises(self):
        with pytest.raises(ValueError, match="slug"):
            class BadTask(FinetuningTask):
                name = "Bad Task"
                slug = None
                description = "Missing slug"
                model_class = AutoModelForCausalLM

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            class BadTask(FinetuningTask):
                name = None
                slug = "bad_task_no_name_1"
                description = "Missing name"
                model_class = AutoModelForCausalLM

    def test_missing_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            class BadTask(FinetuningTask):
                name = "Bad Task"
                slug = "bad_task_no_desc_1"
                description = None
                model_class = AutoModelForCausalLM

    def test_missing_model_class_raises(self):
        with pytest.raises(ValueError, match="model_class"):
            class BadTask(FinetuningTask):
                name = "Bad Task"
                slug = "bad_task_no_cls_1"
                description = "Missing model class"
                model_class = None

    def test_model_class_not_automodel_subclass_raises(self):
        with pytest.raises(ValueError, match="model_class"):
            class BadTask(FinetuningTask):
                name = "Bad Task"
                slug = "bad_task_wrong_cls_1"
                description = "Wrong model class"
                model_class = str

    def test_duplicate_slug_raises(self):
        class FirstTask(FinetuningTask):
            name = "First Task"
            slug = "dup_task_slug_1"
            description = "First"
            model_class = AutoModelForCausalLM

        with pytest.raises(ValueError, match="Duplicate slug"):
            class SecondTask(FinetuningTask):
                name = "Second Task"
                slug = "dup_task_slug_1"
                description = "Second"
                model_class = AutoModelForCausalLM

    def test_causal_lm_registered(self):
        assert "causal_language_modeling" in FinetuningTask._registry
        assert FinetuningTask._registry["causal_language_modeling"] is CausalLM

    def test_causal_lm_model_class_is_auto_model_for_causal_lm(self):
        assert CausalLM.model_class is AutoModelForCausalLM

    def test_causal_lm_has_required_attrs(self):
        assert isinstance(CausalLM.name, str) and CausalLM.name
        assert isinstance(CausalLM.slug, str) and CausalLM.slug
        assert isinstance(CausalLM.description, str) and CausalLM.description
