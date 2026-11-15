from platformdirs import user_data_dir, user_config_dir
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import sqlite3
import re

app_name = "ModelForge"
app_author = "ForgeOpus"
data_dir = Path(user_data_dir(app_name, app_author))
config_dir = Path(user_config_dir(app_name, app_author))

class ModelMetadata(BaseModel):
    id: Optional[int] = Field(None, description="The unique identifier for the model.")
    name: str = Field(..., description="The display name of the model.")
    task_slug: str = Field(..., description="The slug identifier for the fine-tuning task associated with this model.")
    description: Optional[str] = Field(None, description="A detailed description of the model, including its use cases and requirements.")
    model_path: str = Field(..., description="The path where the model is stored.")
    pushed_to_hub: bool = Field(..., description="Indicates whether the model has been pushed to Hugging Face Hub.")
    created_at: str = Field(..., description="The date and time when the model was created.")

    @field_validator('id', mode='before')
    def validate_id(cls, value):
        if value is None:
            return None
        if not isinstance(value, int):
            raise ValueError("ID must be an integer")
        return value
    
    @field_validator('name', 'task_slug', 'model_path', mode='before')
    def validate_non_empty_string(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("This field must be a non-empty string")
        return value.strip()
    
    @field_validator('model_path', mode='before')
    def validate_model_path(cls, value):
        value = value.strip()
        global data_dir
        parent_dir = Path(value).parent
        if not parent_dir == data_dir:
            raise ValueError("Model path must be within the ModelForge data directory")
        return value
    
    @field_validator('created_at', mode='before')
    def validate_created_at(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Created at must be a non-empty string")
        value = value.strip()
        # Accept dates in YYYY-MM-DD or ISO 8601 datetime formats (e.g., 2023-05-01 or 2023-05-01T13:45:30Z)
        iso_regex = re.compile(r"^\d{4}-\d{2}-\d{2}([T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$")
        if not iso_regex.match(value):
            raise ValueError("Created at must be a valid date (YYYY-MM-DD) or ISO 8601 datetime")
        return value

    @field_validator('pushed_to_hub', mode='before')
    def validate_pushed_to_hub(cls, value):
        if not isinstance(value, bool):
            raise ValueError("Pushed to hub must be a boolean value")
        return value


class DatabaseManager:

    _instance = None


    def _initialize(self) -> None:
        """
        Initializes the DatabaseManager by creating the necessary directories and setting up the SQLite database.
        This method is called only once when the singleton instance is created. It ensures that the data
        and config directories exist, and then creates a connection to the SQLite database located in the data directory.

        Args:
            None
        Returns:
            None
        """
        global data_dir, config_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(data_dir / "modelforge.db")
        self.cursor = self.connection.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                task_slug TEXT NOT NULL,
                description TEXT,
                model_path TEXT NOT NULL,
                pushed_to_hub BOOLEAN NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        self._instance = self


    def __new__(cls) -> "DatabaseManager":
        """Implements the singleton pattern to ensure only one instance of DatabaseManager exists.
        Args:
            None
        Returns:
            DatabaseManager: The singleton instance of the DatabaseManager class.
        """
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    
    def save_model(self, model_metadata: ModelMetadata) -> int:
        """Saves a model's metadata to the database. If the model already has an ID, it updates the existing record; otherwise, it creates a new record.
        Args:
            model_metadata (ModelMetadata): An instance of the ModelMetadata class containing the details of the model to be saved.
        Returns:
            int: The ID of the saved model in the database. If the model was updated,
            it returns the existing ID; if a new model was created, it returns the newly generated ID.
        Raises:
            ValueError: If the provided model metadata is invalid or if there are issues with the database operation.
        """

        if model_metadata.id is not None:
            self.cursor.execute('''
                UPDATE models
                SET name = ?, task_slug = ?, description = ?, model_path = ?, pushed_to_hub = ?, created_at = ?
                WHERE id = ?
            ''', (
                model_metadata.name,
                model_metadata.task_slug,
                model_metadata.description,
                model_metadata.model_path,
                model_metadata.pushed_to_hub,
                model_metadata.created_at,
                model_metadata.id
            ))
            self.connection.commit()
            return model_metadata.id
        else:
            if model_metadata.created_at is None or model_metadata.created_at.strip() == "":
                
                model_metadata.created_at = datetime.utcnow().isoformat()
            else:
                # Parse and compare dates without microseconds to avoid precision issues
                created_dt = datetime.fromisoformat(model_metadata.created_at.replace('Z', '+00:00'))
                now_dt = datetime.utcnow()
                if created_dt.replace(microsecond=0) < now_dt.replace(microsecond=0):
                    raise ValueError("Created at cannot be in the past")
            
            if model_metadata.id is not None:
                raise ValueError("ID must be None for new models")

            self.cursor.execute('''
                INSERT INTO models (name, task_slug, description, model_path, pushed_to_hub, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_metadata.name,
                model_metadata.task_slug,
                model_metadata.description,
                model_metadata.model_path,
                model_metadata.pushed_to_hub,
                model_metadata.created_at
            ))
            self.connection.commit()
            return self.cursor.lastrowid
        

    def list_all_models(self, limit: int = 100) -> List[ModelMetadata]:
        """Retrieves a list of all models stored in the database, ordered by creation date in descending order. The number of models returned can be limited by specifying the 'limit' parameter.
        Args:
            limit (int, optional): The maximum number of models to retrieve. Defaults to 100. Put -1 to retrieve all models without limit (potentially slow).
        Returns:
            List[ModelMetadata]: A list of ModelMetadata instances representing the models stored in the database.
        Raises:
            ValueError: If the provided limit is not a positive integer or if there are issues with
            the database operation.
        """
        if not isinstance(limit, int) or (limit <= 0 and limit != -1):
            raise ValueError("Limit must be a positive integer or -1 for no limit")

        if limit == -1:
            self.cursor.execute('''
                SELECT id, name, task_slug, description, model_path, pushed_to_hub, created_at
                FROM models
                ORDER BY created_at DESC
            ''')
        else:
            parameters = (limit,)
            self.cursor.execute('''
                SELECT id, name, task_slug, description, model_path, pushed_to_hub, created_at
                FROM models
                ORDER BY created_at DESC
                LIMIT ?
                ''', parameters
            )
        
        rows = self.cursor.fetchall()
        return [ModelMetadata(
            id=row[0],
            name=row[1],
            task_slug=row[2],
            description=row[3],
            model_path=row[4],
            pushed_to_hub=bool(row[5]),
            created_at=row[6]
        ) for row in rows]

    def delete_model(self, model_id: int) -> Optional[bool]:
        """Deletes a model from the database based on its unique identifier (ID).
        Args:
            model_id (int): The unique identifier of the model to be deleted.
        Returns:
            Optional[bool]: Returns True if the model was successfully deleted, or None if no model with the specified ID was found.
        Raises:
            ValueError: If the provided model ID is not a positive integer or if there are issues with the database operation.
        """

        if not isinstance(model_id, int) or model_id <= 0:
            raise ValueError("Model ID must be a positive integer")
        self.cursor.execute('DELETE FROM models WHERE id = ?', (model_id,))
        self.connection.commit()
        if self.cursor.rowcount == 0:
            raise ValueError(f"No model found with ID {model_id}")
        else:
            return True


class FinetuneLedger:
    """Tracks every completed finetune run."""

    _instance = None

    def _initialize(self) -> None:
        global data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(data_dir / "modelforge.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS finetunes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                model_id    TEXT NOT NULL,
                use_case    TEXT NOT NULL,
                task_slug   TEXT NOT NULL,
                strategy    TEXT NOT NULL,
                peft        TEXT NOT NULL,
                dataset     TEXT NOT NULL,
                output_dir  TEXT NOT NULL,
                config_json TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def __new__(cls) -> "FinetuneLedger":
        if cls._instance is None:
            cls._instance = super(FinetuneLedger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def record(self, model_id: str, use_case: str, task_slug: str, strategy: str,
               peft: str, dataset: str, output_dir: str, config_json: str) -> int:
        """Insert a finetune run record. Returns the new row id."""
        import json
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        self.cursor.execute(
            '''INSERT INTO finetunes
               (timestamp, model_id, use_case, task_slug, strategy, peft, dataset, output_dir, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (timestamp, model_id, use_case, task_slug, strategy, peft, dataset, output_dir, config_json)
        )
        self.connection.commit()
        return self.cursor.lastrowid

    def list_runs(self, limit: int = 100) -> list[dict]:
        """Return all runs ordered by timestamp descending."""
        self.cursor.execute(
            '''SELECT id, timestamp, model_id, use_case, task_slug, strategy, peft, dataset, output_dir, config_json
               FROM finetunes ORDER BY timestamp DESC LIMIT ?''',
            (limit,)
        )
        cols = ["id", "timestamp", "model_id", "use_case", "task_slug", "strategy", "peft", "dataset", "output_dir", "config_json"]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def get_run(self, run_id: int) -> dict | None:
        """Return a single run by id, or None if not found."""
        self.cursor.execute(
            '''SELECT id, timestamp, model_id, use_case, task_slug, strategy, peft, dataset, output_dir, config_json
               FROM finetunes WHERE id = ?''',
            (run_id,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        cols = ["id", "timestamp", "model_id", "use_case", "task_slug", "strategy", "peft", "dataset", "output_dir", "config_json"]
        return dict(zip(cols, row))


class WebRunLedger:
    """Durable run and checkpoint metadata for web/CLI/notebook workflow surfaces."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        global data_dir
        if db_path is None:
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = data_dir / "modelforge.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_runs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    status              TEXT NOT NULL,
                    model_id            TEXT NOT NULL,
                    use_case            TEXT NOT NULL,
                    objective           TEXT NOT NULL,
                    task_slug           TEXT,
                    strategy            TEXT,
                    peft                TEXT,
                    dataset             TEXT NOT NULL,
                    output_dir          TEXT NOT NULL,
                    config_json         TEXT NOT NULL DEFAULT '{}',
                    tensorboard_log_dir TEXT,
                    error               TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_checkpoints (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id     INTEGER NOT NULL,
                    path       TEXT NOT NULL,
                    step       INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, path),
                    FOREIGN KEY(run_id) REFERENCES web_runs(id)
                )
                """
            )
            # Event history, written once when a run reaches a terminal state.
            # The live buffer lives in memory; without this, restarting
            # ModelForge would leave every past run with an empty timeline.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_run_events (
                    run_id      INTEGER PRIMARY KEY,
                    events_json TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES web_runs(id)
                )
                """
            )
            # Trained artifacts. Deliberately separate from the legacy `models`
            # table, whose ModelMetadata requires model_path to sit directly
            # inside the ModelForge data directory -- a user's chosen
            # output_dir almost never does.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_models (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        INTEGER,
                    name          TEXT NOT NULL,
                    base_model    TEXT NOT NULL,
                    task_slug     TEXT,
                    use_case      TEXT,
                    peft_approach TEXT,
                    model_path    TEXT NOT NULL UNIQUE,
                    size_bytes    INTEGER,
                    pushed_to_hub INTEGER NOT NULL DEFAULT 0,
                    hub_repo_id   TEXT,
                    created_at    TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES web_runs(id)
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(web_runs)")}
            if "launch_json" not in columns:
                conn.execute("ALTER TABLE web_runs ADD COLUMN launch_json TEXT NOT NULL DEFAULT '{}'")
            conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    def create_run(
        self,
        *,
        model_id: str,
        use_case: str,
        objective: str,
        dataset: str,
        output_dir: str,
        tensorboard_log_dir: str | None = None,
        config_json: str = "{}",
    ) -> int:
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO web_runs
                (status, model_id, use_case, objective, dataset, output_dir,
                 config_json, tensorboard_log_dir, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "queued",
                    model_id,
                    use_case,
                    objective,
                    dataset,
                    output_dir,
                    config_json,
                    tensorboard_log_dir,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_run(self, run_id: int, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = self._now()
        allowed = {
            "launch_json",
            "status",
            "task_slug",
            "strategy",
            "peft",
            "config_json",
            "tensorboard_log_dir",
            "error",
            "updated_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [run_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE web_runs SET {assignments} WHERE id = ?", params)
            conn.commit()

    def record_checkpoint(self, run_id: int, path: str, step: int | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO web_checkpoints (run_id, path, step, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, path, step, self._now()),
            )
            conn.commit()

    def mark_interrupted_runs(self) -> int:
        """Mark non-terminal runs from a previous process as failed on startup."""
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE web_runs
                SET status = 'failed',
                    error = COALESCE(error, 'ModelForge stopped before this run reached a terminal state.'),
                    updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )
            conn.commit()
            return cursor.rowcount

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM web_runs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM web_runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def save_run_events(self, run_id: int, events_json: str) -> None:
        """Archive a finished run's timeline so it survives a restart."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_run_events (run_id, events_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    events_json = excluded.events_json,
                    created_at = excluded.created_at
                """,
                (run_id, events_json, self._now()),
            )
            conn.commit()

    def load_run_events(self, run_id: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT events_json FROM web_run_events WHERE run_id = ?", (run_id,)
            ).fetchone()
            return row["events_json"] if row else None

    # ------------------------------------------------------------------
    # Trained artifacts
    # ------------------------------------------------------------------

    def record_model(
        self,
        *,
        run_id: int | None,
        name: str,
        base_model: str,
        model_path: str,
        task_slug: str | None = None,
        use_case: str | None = None,
        peft_approach: str | None = None,
        size_bytes: int | None = None,
    ) -> int:
        """Register a finished artifact. Re-registering the same path updates it."""
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO web_models
                    (run_id, name, base_model, task_slug, use_case, peft_approach,
                     model_path, size_bytes, pushed_to_hub, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(model_path) DO UPDATE SET
                    run_id = excluded.run_id,
                    name = excluded.name,
                    base_model = excluded.base_model,
                    task_slug = excluded.task_slug,
                    use_case = excluded.use_case,
                    peft_approach = excluded.peft_approach,
                    size_bytes = excluded.size_bytes
                """,
                (run_id, name, base_model, task_slug, use_case, peft_approach,
                 model_path, size_bytes, now),
            )
            conn.commit()
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = conn.execute(
                "SELECT id FROM web_models WHERE model_path = ?", (model_path,)
            ).fetchone()
            return int(row["id"]) if row else 0

    @staticmethod
    def _model_row(row: sqlite3.Row) -> dict:
        """SQLite stores booleans as integers; return a real bool."""
        record = dict(row)
        record["pushed_to_hub"] = bool(record.get("pushed_to_hub"))
        return record

    def list_models(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM web_models ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._model_row(row) for row in rows]

    def get_model(self, model_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM web_models WHERE id = ?", (model_id,)).fetchone()
            return self._model_row(row) if row else None

    def delete_model(self, model_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM web_models WHERE id = ?", (model_id,))
            conn.commit()
            return cursor.rowcount > 0

    def mark_model_pushed(self, model_id: int, repo_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE web_models SET pushed_to_hub = 1, hub_repo_id = ? WHERE id = ?",
                (repo_id, model_id),
            )
            conn.commit()

    def list_checkpoints(self, run_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM web_checkpoints
                WHERE run_id = ?
                ORDER BY COALESCE(step, 0) DESC, created_at DESC
                """,
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]
