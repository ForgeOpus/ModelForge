import os
import sqlite3
from datetime import datetime
import traceback
import shutil
from typing import Any

class DatabaseManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: os.PathLike | str) -> None:
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            open(self.db_path, 'w').close()
        self.conn = None
        self.cursor = None
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

            print("Initializing database...")

            # Create fine-tuned models table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fine_tuned_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                base_model TEXT NOT NULL,
                task TEXT NOT NULL,
                description TEXT,
                creation_date TEXT NOT NULL,
                model_path TEXT NOT NULL
            )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            traceback.print_exc()
        finally:
            self.kill_connection()

    def add_model(self, model_data : dict[str: Any]) -> int | None:
        """Add a new fine-tuned model to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
            INSERT INTO fine_tuned_models 
            (model_name, base_model, task, description, creation_date, 
            model_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_data['model_name'],
                model_data['base_model'],
                model_data['task'],
                model_data.get('description', ''),
                model_data.get('creation_date', datetime.now().isoformat()),
                model_data['model_path'],
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding model to database: {e}")
            traceback.print_exc()
            return None
        finally:
            self.kill_connection()

    def get_all_models(self) -> list[dict] | None:
        """Get all fine-tuned models"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()

            self.cursor.execute('''
            SELECT * FROM fine_tuned_models
            ORDER BY creation_date DESC
            ''')

            rows = self.cursor.fetchall()
            models = []
            for row in rows:
                model = dict(row)
                models.append(model)
            return models
        except sqlite3.Error as e:
            print(f"Error retrieving models: {e}")
            return []
        finally:
            self.kill_connection()

    def get_model_by_id(self, model_id) -> dict | None:
        """Get a specific model by ID"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()

            self.cursor.execute('SELECT * FROM fine_tuned_models WHERE id = ?', (model_id,))
            row = self.cursor.fetchone()

            if row:
                model = dict(row)
                return model
            return None
        except sqlite3.Error as e:
            print(f"Error retrieving model: {e}")
            return None
        finally:
            self.kill_connection()

    def delete_model(self, model_id) -> bool:
        """Delete a model from DB and directory"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            model_path = self.get_model_by_id(model_id)['model_path']
            if model_path and os.path.exists(model_path):
                shutil.rmtree(model_path, ignore_errors=True)
                print(f"Deleted model files at {model_path}")
            self.cursor.execute('DELETE FROM fine_tuned_models WHERE id = ?', (model_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting model: {e}")
            traceback.print_exc()
            return False
        finally:
            self.kill_connection()

    def kill_connection(self) -> None:
        if self.conn:
            self.conn.close()
            del self.cursor