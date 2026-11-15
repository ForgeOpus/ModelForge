import pytest
from datetime import datetime
from pathlib import Path
from utilities.database_manager import DatabaseManager, ModelMetadata


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    import utilities.database_manager as dm
    monkeypatch.setattr(dm, "data_dir", tmp_path)
    monkeypatch.setattr(dm, "config_dir", tmp_path)
    DatabaseManager._instance = None
    yield
    if DatabaseManager._instance is not None:
        DatabaseManager._instance.connection.close()
    DatabaseManager._instance = None


def make_metadata(tmp_path, **overrides):
    """Helper to build valid ModelMetadata, with model_path inside tmp_path."""
    defaults = dict(
        name="Test Model",
        task_slug="causal_language_modeling",
        description="A test model",
        model_path=str(tmp_path / "test_model"),
        pushed_to_hub=False,
        created_at=datetime.utcnow().isoformat(),
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


# ---------------------------------------------------------------------------
# ModelMetadata validation
# ---------------------------------------------------------------------------

class TestModelMetadataValidation:
    def test_valid_metadata_constructs(self, tmp_path):
        m = make_metadata(tmp_path)
        assert m.name == "Test Model"

    def test_id_none_is_allowed(self, tmp_path):
        m = make_metadata(tmp_path, id=None)
        assert m.id is None

    def test_id_non_int_raises(self, tmp_path):
        with pytest.raises(Exception, match="integer"):
            make_metadata(tmp_path, id="abc")

    def test_name_whitespace_only_raises(self, tmp_path):
        with pytest.raises(Exception):
            make_metadata(tmp_path, name="   ")

    def test_task_slug_empty_raises(self, tmp_path):
        with pytest.raises(Exception):
            make_metadata(tmp_path, task_slug="")

    def test_model_path_outside_data_dir_raises(self, tmp_path):
        with pytest.raises(Exception, match="ModelForge data directory"):
            make_metadata(tmp_path, model_path="/some/other/path/model")

    def test_model_path_whitespace_stripped(self, tmp_path):
        path = str(tmp_path / "test_model")
        m = make_metadata(tmp_path, model_path=f"  {path}  ")
        assert m.model_path == path

    def test_created_at_iso_date_accepted(self, tmp_path):
        m = make_metadata(tmp_path, created_at="2024-01-15")
        assert m.created_at == "2024-01-15"

    def test_created_at_iso_datetime_accepted(self, tmp_path):
        m = make_metadata(tmp_path, created_at="2024-01-15T10:30:00")
        assert m.created_at == "2024-01-15T10:30:00"

    def test_created_at_invalid_format_raises(self, tmp_path):
        with pytest.raises(Exception, match="ISO 8601"):
            make_metadata(tmp_path, created_at="01/15/2024")

    def test_pushed_to_hub_int_raises(self, tmp_path):
        with pytest.raises(Exception, match="boolean"):
            make_metadata(tmp_path, pushed_to_hub=1)


# ---------------------------------------------------------------------------
# DatabaseManager CRUD
# ---------------------------------------------------------------------------

class TestDatabaseManagerCRUD:
    def test_singleton_two_calls_same_instance(self):
        db1 = DatabaseManager()
        db2 = DatabaseManager()
        assert db1 is db2

    def test_save_new_model_returns_positive_int_id(self, tmp_path):
        db = DatabaseManager()
        m = make_metadata(tmp_path)
        result = db.save_model(m)
        assert isinstance(result, int)
        assert result > 0

    def test_save_model_update_changes_name(self, tmp_path):
        db = DatabaseManager()
        m = make_metadata(tmp_path)
        model_id = db.save_model(m)
        updated = make_metadata(tmp_path, id=model_id, name="Updated Name")
        db.save_model(updated)
        models = db.list_all_models()
        assert any(mdl.name == "Updated Name" for mdl in models)

    def test_list_all_models_empty_db(self):
        db = DatabaseManager()
        assert db.list_all_models() == []

    def test_list_all_models_returns_all_inserted(self, tmp_path):
        db = DatabaseManager()
        for i in range(3):
            db.save_model(make_metadata(tmp_path, name=f"Model {i}"))
        assert len(db.list_all_models()) == 3

    def test_list_all_models_limit_applied(self, tmp_path):
        db = DatabaseManager()
        for i in range(5):
            db.save_model(make_metadata(tmp_path, name=f"Model {i}"))
        assert len(db.list_all_models(limit=2)) == 2

    def test_list_all_models_limit_minus1_all(self, tmp_path):
        db = DatabaseManager()
        for i in range(3):
            db.save_model(make_metadata(tmp_path, name=f"Model {i}"))
        assert len(db.list_all_models(limit=-1)) == 3

    def test_list_all_models_invalid_limit_raises(self):
        db = DatabaseManager()
        with pytest.raises(ValueError):
            db.list_all_models(limit=0)

    def test_list_all_models_returns_model_metadata_instances(self, tmp_path):
        db = DatabaseManager()
        db.save_model(make_metadata(tmp_path))
        results = db.list_all_models()
        assert all(isinstance(m, ModelMetadata) for m in results)

    def test_delete_model_returns_true(self, tmp_path):
        db = DatabaseManager()
        model_id = db.save_model(make_metadata(tmp_path))
        assert db.delete_model(model_id) is True

    def test_delete_model_removes_from_list(self, tmp_path):
        db = DatabaseManager()
        model_id = db.save_model(make_metadata(tmp_path))
        db.delete_model(model_id)
        assert db.list_all_models() == []

    def test_delete_nonexistent_raises(self):
        db = DatabaseManager()
        with pytest.raises(ValueError):
            db.delete_model(9999)

    def test_delete_invalid_id_raises(self):
        db = DatabaseManager()
        with pytest.raises(ValueError):
            db.delete_model(-1)

    def test_save_new_model_past_timestamp_raises(self, tmp_path):
        db = DatabaseManager()
        m = make_metadata(tmp_path, created_at="2020-01-01T00:00:00")
        with pytest.raises(ValueError, match="past"):
            db.save_model(m)
