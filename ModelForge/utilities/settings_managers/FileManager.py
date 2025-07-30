import _io

from platformdirs import user_data_dir
import os

class FileManager:
    """
    A singleton class to manage non-database file operations like uploading dataset files,
    creating base directories, and fetching dataset files.
    """

    dirs_base = user_data_dir(appname="ModelForge")
    _instance = None
    _default_dirs = {
        "datasets": os.path.abspath(os.path.join(dirs_base, "datasets")),
        "models": os.path.abspath(os.path.join(dirs_base, "models")),
        "logs": os.path.abspath(os.path.join(dirs_base, "logs")),
        "database": os.path.abspath(os.path.join(dirs_base, "database")),
        "model_checkpoints": os.path.abspath(os.path.join(dirs_base, "model_checkpoints")),
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FileManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            print("Initializing file manager...")
            print(f"Setting default directories: {self._default_dirs}")
            for dir_name, dir_path in self._default_dirs.items():
                self.validate_or_create_dirs(dir_path)
            print("File manager initialized.")

    @classmethod
    def get_instance(cls):
        return cls.__new__(cls)

    @classmethod
    def validate_or_create_dirs(cls, check_path: str) -> str:
        os.makedirs(check_path, exist_ok=True)
        return check_path

    @classmethod
    def validate_or_create_file(cls, check_path: str) -> str:
        if not os.path.exists(os.path.abspath(check_path)):
            os.makedirs(os.path.dirname(os.path.abspath(check_path)), exist_ok=True)
            open(os.path.abspath(check_path), 'w').close()
        return check_path

    @classmethod
    def save_file(cls, content:bytes, file_path: str) -> str | None:
        try:
            file_dir = os.path.dirname(file_path)
            cls.validate_or_create_dirs(os.path.abspath(file_dir))
            with open(file_path, 'wb') as f:
                f.write(content)
            return file_path
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    @classmethod
    def return_default_dirs(cls) -> dict[str, str]:
        return cls._default_dirs