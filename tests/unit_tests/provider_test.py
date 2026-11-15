import pytest
from providers.provider import Provider
from providers.provider_factory import ProviderFactory


@pytest.fixture(autouse=True)
def clean_provider_registry():
    original = dict(Provider._registry)
    yield
    Provider._registry.clear()
    Provider._registry.update(original)


class TestProviderRegistry:
    def test_registry_starts_empty(self):
        # No built-in providers defined yet
        assert Provider._registry == {}

    def test_valid_provider_subclass_registers(self):
        class MyProvider(Provider):
            name = "My Provider"
            slug = "my_provider"
            description = "Test provider"

        assert "my_provider" in Provider._registry
        assert Provider._registry["my_provider"] is MyProvider

    def test_provider_subclass_non_string_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            class BadProvider(Provider):
                name = 42
                slug = "bad_provider_1"
                description = "Bad provider"

    def test_provider_subclass_duplicate_slug_raises(self):
        class FirstProvider(Provider):
            name = "First"
            slug = "first_provider"
            description = "First"

        with pytest.raises(ValueError):
            class SecondProvider(Provider):
                name = "Second"
                slug = "first_provider"
                description = "Second"

    def test_get_available_providers_empty(self):
        assert ProviderFactory.get_available_providers() == []

    def test_get_provider_unknown_slug_raises_value_error(self):
        with pytest.raises(ValueError):
            ProviderFactory.get_provider("unknown")

    def test_get_provider_known_slug_returns_class(self):
        class FakeClass:
            pass
        Provider._registry["test_slug"] = FakeClass
        result = ProviderFactory.get_provider("test_slug")
        assert result is FakeClass

    def test_clear_registry_via_factory_empties_registry(self):
        class FakeClass:
            pass
        Provider._registry["test_slug"] = FakeClass
        ProviderFactory.clear_registry()
        assert Provider._registry == {}
