import pytest
from unittest.mock import patch, MagicMock, call
from dataset_manager.dataset_factory import DatasetFactory

MOCK_DS = MagicMock()


class RenameableDataset:
    def __init__(self, columns):
        self.column_names = columns
        self.features = {column: object() for column in columns}
        self.renamed = None

    def rename_columns(self, columns):
        self.renamed = columns
        return self


class TestNormalizeColumns:
    def test_renames_documented_aliases(self):
        dataset = RenameableDataset(["document", "summary"])

        result = DatasetFactory.normalize_columns(
            dataset, {"document": "prompt", "summary": "completion"}
        )

        assert result is dataset
        assert dataset.renamed == {"document": "prompt", "summary": "completion"}

    def test_accepts_already_normalized_columns(self):
        dataset = RenameableDataset(["prompt", "completion"])

        assert DatasetFactory.normalize_columns(
            dataset, {"document": "prompt", "summary": "completion"}
        ) is dataset
        assert dataset.renamed is None

    def test_rejects_ambiguous_alias_and_canonical_column(self):
        dataset = RenameableDataset(["document", "prompt", "summary"])

        with pytest.raises(ValueError, match="both an alias"):
            DatasetFactory.normalize_columns(
                dataset, {"document": "prompt", "summary": "completion"}
            )

    def test_accepts_dpo_implicit_prompt_pairs(self):
        dataset = RenameableDataset(["chosen", "rejected"])

        assert DatasetFactory.normalize_columns(
            dataset,
            {"preferred_response": "chosen", "dispreferred_response": "rejected"},
            {"instruction": "prompt"},
        ) is dataset
        assert dataset.renamed is None

    def test_renames_optional_dpo_prompt_alias_when_present(self):
        dataset = RenameableDataset(["instruction", "preferred_response", "dispreferred_response"])

        result = DatasetFactory.normalize_columns(
            dataset,
            {"preferred_response": "chosen", "dispreferred_response": "rejected"},
            {"instruction": "prompt"},
        )

        assert result is dataset
        assert dataset.renamed == {
            "instruction": "prompt",
            "preferred_response": "chosen",
            "dispreferred_response": "rejected",
        }


# ---------------------------------------------------------------------------
# handle_json
# ---------------------------------------------------------------------------

class TestHandleJson:
    def test_jsonl_succeeds_on_first_try(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text":"hi"}\n{"text":"bye"}\n')
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            result = DatasetFactory.handle_json(str(f))
        m.assert_called_once_with("json", data_files=str(f), streaming=False)
        assert result is MOCK_DS

    def test_json_array_fallback(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('[{"text":"hi"},{"text":"bye"}]')
        with patch("dataset_manager.dataset_factory.load_dataset") as m:
            m.side_effect = [Exception("not jsonl"), MOCK_DS]
            result = DatasetFactory.handle_json(str(f))
        assert result is MOCK_DS
        assert m.call_args_list[1] == call("json", data_files=str(f))

    def test_json_nested_field_detected(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"meta":{},"records":[{"text":"hi"},{"text":"bye"}]}')
        with patch("dataset_manager.dataset_factory.load_dataset") as m:
            m.side_effect = [Exception("not jsonl"), MOCK_DS]
            result = DatasetFactory.handle_json(str(f))
        assert result is MOCK_DS
        assert m.call_args_list[1] == call("json", data_files=str(f), field="records")

    def test_json_multiple_array_fields_tries_in_order(self, tmp_path):
        # First candidate "tags" fails, second "records" succeeds.
        f = tmp_path / "data.json"
        f.write_text('{"tags":[],"records":[{"text":"hi"}]}')
        with patch("dataset_manager.dataset_factory.load_dataset") as m:
            m.side_effect = [
                Exception("not jsonl"),  # initial JSONL attempt
                Exception("bad field"),  # tags
                MOCK_DS,                 # records
            ]
            result = DatasetFactory.handle_json(str(f))
        assert result is MOCK_DS
        assert m.call_args_list[2] == call("json", data_files=str(f), field="records")

    def test_json_no_array_field_raises(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"meta":"value","count":42}')
        with patch("dataset_manager.dataset_factory.load_dataset", side_effect=Exception("not jsonl")):
            with pytest.raises(ValueError, match="No list field found"):
                DatasetFactory.handle_json(str(f))

    def test_unrecognized_structure_raises(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('"just a string"')
        with patch("dataset_manager.dataset_factory.load_dataset", side_effect=Exception("not jsonl")):
            with pytest.raises(ValueError, match="Unrecognized JSON structure"):
                DatasetFactory.handle_json(str(f))

    def test_streaming_re_raises_on_failure(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('[{"text":"hi"}]')
        with patch("dataset_manager.dataset_factory.load_dataset", side_effect=RuntimeError("stream err")):
            with pytest.raises(RuntimeError, match="stream err"):
                DatasetFactory.handle_json(str(f), streaming=True)


# ---------------------------------------------------------------------------
# get_local_dataset
# ---------------------------------------------------------------------------

class TestGetLocalDataset:
    def test_csv(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            result = DatasetFactory.get_local_dataset("csv", "data.csv")
        m.assert_called_once_with("csv", data_files="data.csv", streaming=False)
        assert result is MOCK_DS

    def test_parquet(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            result = DatasetFactory.get_local_dataset("parquet", "data.parquet")
        m.assert_called_once_with("parquet", data_files={"train": "data.parquet"}, streaming=False)
        assert result is MOCK_DS

    def test_arrow(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            result = DatasetFactory.get_local_dataset("arrow", "data.arrow")
        m.assert_called_once_with("arrow", data_files={"train": "data.arrow"}, streaming=False)
        assert result is MOCK_DS

    def test_text(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            result = DatasetFactory.get_local_dataset("text", "data.txt")
        m.assert_called_once_with("text", data_files="data.txt", streaming=False)
        assert result is MOCK_DS

    def test_json_delegates_to_handle_json(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\n')
        with patch.object(DatasetFactory, "handle_json", return_value=MOCK_DS) as m:
            result = DatasetFactory.get_local_dataset("json", str(f))
        m.assert_called_once_with(str(f), streaming=False)
        assert result is MOCK_DS

    def test_streaming_passed_through(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.get_local_dataset("csv", "data.csv", streaming=True)
        m.assert_called_once_with("csv", data_files="data.csv", streaming=True)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported local dataset type"):
            DatasetFactory.get_local_dataset("xml", "data.xml")


# ---------------------------------------------------------------------------
# get_remote_dataset
# ---------------------------------------------------------------------------

class TestGetRemoteDataset:
    def test_basic(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            result = DatasetFactory.get_remote_dataset("squad")
        m.assert_called_once_with("squad", name=None, streaming=False)
        assert result is MOCK_DS

    def test_with_name(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.get_remote_dataset("glue", name="mrpc")
        m.assert_called_once_with("glue", name="mrpc", streaming=False)

    def test_streaming(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.get_remote_dataset("squad", streaming=True)
        m.assert_called_once_with("squad", name=None, streaming=True)

    def test_extra_kwargs_forwarded(self):
        with patch("dataset_manager.dataset_factory.load_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.get_remote_dataset("squad", split="train")
        m.assert_called_once_with("squad", name=None, streaming=False, split="train")

    def test_load_error_re_raises(self):
        with patch("dataset_manager.dataset_factory.load_dataset", side_effect=ConnectionError("no network")):
            with pytest.raises(ConnectionError, match="no network"):
                DatasetFactory.get_remote_dataset("squad")


# ---------------------------------------------------------------------------
# load() — routing logic
# ---------------------------------------------------------------------------

class TestLoad:
    # Extension-based inference
    @pytest.mark.parametrize("filename,expected_type", [
        ("data.csv",     "csv"),
        ("data.json",    "json"),
        ("data.jsonl",   "json"),
        ("data.parquet", "parquet"),
        ("data.arrow",   "arrow"),
        ("data.txt",     "text"),
    ])
    def test_extension_inference(self, filename, expected_type):
        with patch("dataset_manager.dataset_factory.os.path.exists", return_value=True):
            with patch.object(DatasetFactory, "get_local_dataset", return_value=MOCK_DS) as m:
                DatasetFactory.load(filename)
        m.assert_called_once_with(expected_type, filename, streaming=False)

    def test_explicit_type_used_when_provided(self):
        with patch("dataset_manager.dataset_factory.os.path.exists", return_value=True):
            with patch.object(DatasetFactory, "get_local_dataset", return_value=MOCK_DS) as m:
                DatasetFactory.load("data.csv", type_="csv")
        m.assert_called_once_with("csv", "data.csv", streaming=False)

    def test_no_extension_falls_through_to_remote(self):
        with patch.object(DatasetFactory, "get_remote_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.load("squad")
        m.assert_called_once_with("squad", streaming=False)

    # remote=True flag
    def test_remote_true_bypasses_extension(self):
        # Path ends in .json but remote=True should skip local detection.
        with patch.object(DatasetFactory, "get_remote_dataset", return_value=MOCK_DS) as m:
            with patch.object(DatasetFactory, "get_local_dataset") as local_m:
                DatasetFactory.load("dataset.json", remote=True)
        m.assert_called_once_with("dataset.json", streaming=False)
        local_m.assert_not_called()

    def test_remote_true_passes_kwargs(self):
        with patch.object(DatasetFactory, "get_remote_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.load("glue", remote=True, name="mrpc")
        m.assert_called_once_with("glue", streaming=False, name="mrpc")

    def test_remote_true_with_streaming(self):
        with patch.object(DatasetFactory, "get_remote_dataset", return_value=MOCK_DS) as m:
            DatasetFactory.load("squad", remote=True, streaming=True)
        m.assert_called_once_with("squad", streaming=True)

    def test_streaming_forwarded_to_local(self):
        with patch("dataset_manager.dataset_factory.os.path.exists", return_value=True):
            with patch.object(DatasetFactory, "get_local_dataset", return_value=MOCK_DS) as m:
                DatasetFactory.load("data.csv", streaming=True)
        m.assert_called_once_with("csv", "data.csv", streaming=True)
