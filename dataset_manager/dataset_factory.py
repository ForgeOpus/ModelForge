import re
import logging
from datasets import load_dataset, DatasetDict, IterableDataset
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetFactory():

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
    def get_local_dataset(cls, type_: str, path: str, streaming: bool=False) -> Union[DatasetDict, IterableDataset]:
        """Loads a local dataset file based on its type."""
        try:
            dataset: Optional[Union[DatasetDict, IterableDataset]] = None
            
            if type_ == "csv":
                dataset = load_dataset("csv", data_files=path, streaming=streaming)
            elif type_ == "json":
                dataset = cls.handle_json(path, streaming=streaming)
            elif type_ == "parquet":
                dataset = load_dataset("parquet", data_files={'train': path}, streaming=streaming)
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

        if type_ or (isinstance(path, str) and (path.endswith(('.csv', '.json', '.jsonl', '.parquet', '.arrow', '.txt')))):
            # If type not provided but extension is present, infer it
            if not type_ and isinstance(path, str):
                ext = path.split('.')[-1].lower()
                type_ = "json" if ext in ["json", "jsonl"] else ext
                if type_ == "txt": type_ = "text"

            return cls.get_local_dataset(type_ or "json", path, streaming=streaming)

        return cls.get_remote_dataset(path, streaming=streaming, **kwargs)


        
