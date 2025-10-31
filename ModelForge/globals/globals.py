import os

from ..utilities.settings_managers.DBManager import DatabaseManager
from ..utilities.settings_managers.FileManager import FileManager
from ..utilities.hardware_detection.hardware_detector import HardwareDetector
from ..utilities.finetuning.settings_builder import SettingsBuilder

class GlobalSettings:
    _instance = None
    _initialized = False
    file_manager = None
    hardware_detector = None
    settings_builder = None
    settings_cache = None
    finetuning_status = None
    datasets_dir = None
    model_path = None
    db_manager = None
    app_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalSettings, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once to maintain singleton behavior
        if not GlobalSettings._initialized:
            self.file_manager = FileManager()
            self.hardware_detector = HardwareDetector()
            self.settings_builder = SettingsBuilder(None, None, None)
            self.settings_cache = {}
            # NOTE: finetuning_status is accessed from multiple places (callback, background task)
            # without locking. Python's GIL provides basic protection, but be cautious with
            # complex operations. Consider using threading.Lock if race conditions occur.
            self.finetuning_status = {"status": "idle", "progress": 0, "message": ""}
            self.datasets_dir = self.file_manager.return_default_dirs()["datasets"]
            self.model_path = self.file_manager.return_default_dirs()["models"]
            self.db_manager = DatabaseManager(db_path=os.path.join(self.file_manager.return_default_dirs()["database"], "modelforge.sqlite"))
            self.app_name = "ModelForge"
            GlobalSettings._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def clear_settings_cache(self):
        self.settings_cache.clear()

    def reset_finetuning_status(self):
        self.finetuning_status = {"status": "idle", "progress": 0, "message": ""}

    def reset_settings_builder(self):
        self.settings_builder.reset()

    def get_app_name(self):
        return self.app_name

    def get_db_manager(self):
        return self.db_manager

    def get_hardware_detector(self):
        return self.hardware_detector

    def get_settings_builder(self):
        return self.settings_builder

    def get_settings_cache(self):
        return self.settings_cache

    def get_finetuning_status(self):
        return self.finetuning_status

    def get_datasets_dir(self):
        return self.datasets_dir

    def get_model_path(self):
        return self.model_path


