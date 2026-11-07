from datasets import load_dataset, DatasetDict, IterableDataset
from typing import Optional, Union
import logging
import re
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_EXT_PRIORITY = ["parquet", "arrow", "jsonl", "json", "csv", "txt"]
_EXT_NORMALIZE = {"jsonl": "json", "txt": "text"}

class DatasetFactory:

    @staticmethod
    def normalize_columns(
        dataset,
        column_aliases: dict[str, str] | None,
        optional_column_aliases: dict[str, str] | None = None,
    ):
        """Rename a use-case's documented source columns to trainer columns.

        ``DataFormatSpec.column_aliases`` is a source-to-destination map.  Do
        not silently overwrite a canonical column: a dataset containing both
        ``document`` and ``prompt``, for example, is ambiguous and must be
        corrected by its author rather than guessing which content to train on.
        """
        aliases = {**(column_aliases or {}), **(optional_column_aliases or {})}
        optional_sources = set((optional_column_aliases or {}).keys())
        columns = list(getattr(dataset, "column_names", []) or getattr(dataset, "features", {}).keys())
        rename = {source: destination for source, destination in aliases.items() if source in columns and source != destination}
        missing = [
            source
            for source in aliases
            if source not in optional_sources and source not in columns and aliases[source] not in columns
        ]
        if missing:
            raise ValueError(
                "Dataset is missing required column(s): " + ", ".join(sorted(missing)) + "."
            )
        collisions = [
            f"{source!r} -> {destination!r}"
            for source, destination in rename.items()
            if destination in columns
        ]
        if collisions:
            raise ValueError(
                "Dataset contains both an alias and its canonical training column: "
                + ", ".join(collisions) + ". Remove or rename one explicitly."
            )
        return dataset.rename_columns(rename) if rename else dataset

    @classmethod
    def _detect_dir_type(cls, path: str) -> tuple[str, list[str]]:
        """Scan a directory and return (normalized_type, sorted_file_list) for the dominant extension."""
        files_by_ext: dict[str, list[str]] = {}
        for fname in sorted(os.listdir(path)):
            if "." in fname:
                ext = fname.rsplit(".", 1)[-1].lower()
                if ext in set(_EXT_PRIORITY):
                    files_by_ext.setdefault(ext, []).append(os.path.join(path, fname))

        for ext in _EXT_PRIORITY:
            if ext in files_by_ext:
                normalized = _EXT_NORMALIZE.get(ext, ext)
                logger.info(f"Detected {len(files_by_ext[ext])} .{ext} shard(s) in {path}")
                return normalized, files_by_ext[ext]

        raise ValueError(f"No supported dataset files found in directory: {path}")

    @classmethod
    def handle_json(cls, path: str, streaming: bool = False) -> Union[DatasetDict, IterableDataset]:
        """
        Checks whether a passed JSON dataset is of type:
            - JSON
            - JSONL
            - Nested field (auto-detects the largest list field)
        """
        # Try loading as JSONL first (best for streaming)
        try:
            return load_dataset("json", data_files=path, streaming=streaming)
        except Exception as e:
            if streaming:
                logger.error(f"Streaming failed for {path}. JSONL is required for streaming JSON. Error: {e}")
                raise

        # If it fails, it might be a nested JSON or list.
        # Peek at the file header instead of loading the entire file into memory.
        try:
            with open(path, 'rb') as f:
                header = f.read(65536).lstrip()

            if header.startswith(b'['):
                return load_dataset("json", data_files=path)

            if header.startswith(b'{'):
                # Regex finds top-level "key": [ patterns; false positives are
                # filtered out by trying each candidate with load_dataset.
                matches = re.findall(rb'"([^"]+)"\s*:\s*\[', header)
                if not matches:
                    raise ValueError(f"No list field found in JSON object at {path}")

                for field in (m.decode('utf-8') for m in matches):
                    try:
                        result = load_dataset("json", data_files=path, field=field)
                        logger.info(f"Auto-detected nested field '{field}' in {path}")
                        return result
                    except Exception:
                        continue

                raise ValueError(f"Could not load JSON with any detected array field in {path}")

            raise ValueError(f"Unrecognized JSON structure in {path}")

        except Exception as e:
            logger.error(f"Failed to handle JSON file {path}: {e}")
            raise ValueError(f"Invalid JSON structure for dataset loading: {e}")

    @classmethod
    def get_local_dataset(cls, type_: str, path: Union[str, list[str]], streaming: bool = False) -> Union[DatasetDict, IterableDataset]:
        """Loads a local dataset file (or list of shards) based on its type."""
        try:
            dataset: Optional[Union[DatasetDict, IterableDataset]] = None
            is_sharded = isinstance(path, list)

            if type_ == "csv":
                dataset = load_dataset("csv", data_files=path, streaming=streaming)
            elif type_ == "json":
                # Shard files are always flat JSONL — skip nested-field detection
                if is_sharded:
                    dataset = load_dataset("json", data_files=path, streaming=streaming)
                else:
                    dataset = cls.handle_json(path, streaming=streaming)
            elif type_ == "parquet":
                dataset = load_dataset("parquet", data_files={"train": path}, streaming=streaming)
            elif type_ == "arrow":
                dataset = load_dataset("arrow", data_files={"train": path}, streaming=streaming)
            elif type_ == "text":
                dataset = load_dataset("text", data_files=path, streaming=streaming)
            else:
                raise ValueError(f"Unsupported local dataset type: {type_}")

            return dataset
        except Exception as e:
            logger.error(f"Error loading local {type_} dataset from {path}: {e}")
            raise

    @classmethod
    def get_remote_dataset(cls, path: str, name: Optional[str] = None, streaming: bool = False, **kwargs) -> Union[DatasetDict, IterableDataset]:
        """Loads a dataset from the Hugging Face Hub."""
        try:
            return load_dataset(path, name=name, streaming=streaming, **kwargs)
        except Exception as e:
            logger.error(f"Error loading remote dataset {path}: {e}")
            raise

    @classmethod
    def load(cls, path: str, type_: Optional[str] = None, streaming: bool = False, remote: bool = False, **kwargs) -> Union[DatasetDict, IterableDataset]:
        """
        Unified entry point to load any dataset (local or remote).
        If type_ is provided, it assumes a local file.
        Otherwise, it attempts to load from the HF Hub.
        Set remote=True to skip local detection and load directly from the HF Hub.
        """
        if remote:
            return cls.get_remote_dataset(path, streaming=streaming, **kwargs)

        if os.path.isdir(path):
            detected_type, shard_files = cls._detect_dir_type(path)
            return cls.get_local_dataset(detected_type, shard_files, streaming=streaming)

        if type_ or (isinstance(path, str) and (path.endswith(('.csv', '.json', '.jsonl', '.parquet', '.arrow', '.txt')))):
            if not os.path.exists(path):
                logger.error(f"Local dataset file not found: {path}")
                raise FileNotFoundError(f"Local dataset file not found: {path}")
                
            # If type not provided but extension is present, infer it
            if not type_ and isinstance(path, str):
                ext = path.split('.')[-1].lower()
                type_ = "json" if ext in ["json", "jsonl"] else ext
                if type_ == "txt": type_ = "text"

            return cls.get_local_dataset(type_ or "json", path, streaming=streaming)

        return cls.get_remote_dataset(path, streaming=streaming, **kwargs)


        
