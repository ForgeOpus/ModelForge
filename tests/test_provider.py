import pytest

from ModelForge.utilities.finetuning.settings_builder import SettingsBuilder
from ModelForge.utilities.finetuning.providers import ProviderRegistry


def test_settings_builder_provider_persistence():
    sb = SettingsBuilder(task="text-generation", model_name="dummy/model", compute_profile="low_end")
    available = ProviderRegistry.list_available()
    assert available, "No providers registered; expected at least one (e.g., 'huggingface')."
    provider = "huggingface" if "huggingface" in available else available[0]
    sb.set_settings({"provider": provider})
    assert sb.provider == provider
    settings = sb.get_settings()
    assert settings["provider"] == provider


def test_settings_builder_invalid_provider():
    sb = SettingsBuilder(task="text-generation", model_name="dummy/model", compute_profile="low_end")
    with pytest.raises(ValueError):
        sb.set_settings({"provider": "invalid_provider_name_xyz"})


def test_settings_builder_default_provider():
    sb = SettingsBuilder(task="text-generation", model_name="dummy/model", compute_profile="low_end")
    # Call set_settings without provider to ensure default is applied
    sb.set_settings({"learning_rate": 0.0002})
    assert sb.provider == "huggingface"
    assert sb.get_settings()["provider"] == "huggingface"
