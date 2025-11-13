"""
Dependency injection for ModelForge.
Provides factory functions for services and managers.
"""
import os

from .database.database_manager import DatabaseManager
from .services.training_service import TrainingService
from .services.model_service import ModelService
from .services.hardware_service import HardwareService
from .utilities.settings_managers.FileManager import FileManager
from .logging_config import logger


# Global instances (singleton pattern for stateless services)
_db_manager = None
_file_manager = None
_training_service = None
_model_service = None
_hardware_service = None

# Session cache for storing temporary user selections
_session_cache = {}


def get_file_manager() -> FileManager:
    """
    Get FileManager instance.

    Returns:
        FileManager instance
    """
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
        logger.info("FileManager initialized")
    return _file_manager


def get_db_manager() -> DatabaseManager:
    """
    Get DatabaseManager instance.

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        file_manager = get_file_manager()
        default_dirs = file_manager.return_default_dirs()
        db_path = os.path.join(default_dirs["database"], "modelforge.sqlite")
        _db_manager = DatabaseManager(db_path)
        logger.info("DatabaseManager initialized")
    return _db_manager


def get_training_service() -> TrainingService:
    """
    Get TrainingService instance.

    Returns:
        TrainingService instance
    """
    global _training_service
    if _training_service is None:
        db_manager = get_db_manager()
        file_manager = get_file_manager()
        _training_service = TrainingService(
            db_manager=db_manager,
            file_manager=file_manager,
        )
        logger.info("TrainingService initialized")
    return _training_service


def get_model_service() -> ModelService:
    """
    Get ModelService instance.

    Returns:
        ModelService instance
    """
    global _model_service
    if _model_service is None:
        db_manager = get_db_manager()
        _model_service = ModelService(db_manager=db_manager)
        logger.info("ModelService initialized")
    return _model_service


def get_hardware_service() -> HardwareService:
    """
    Get HardwareService instance.

    Returns:
        HardwareService instance
    """
    global _hardware_service
    if _hardware_service is None:
        _hardware_service = HardwareService()
        logger.info("HardwareService initialized")
    return _hardware_service


def get_session_data(key: str = None):
    """
    Get session data from cache.

    Args:
        key: Specific key to retrieve. If None, returns entire cache.

    Returns:
        Session data for key, or entire cache, or None if not found
    """
    global _session_cache
    if key is None:
        return _session_cache.copy()
    return _session_cache.get(key)


def update_session_data(key: str, value):
    """
    Update session data in cache.

    Args:
        key: Key to store
        value: Value to store
    """
    global _session_cache
    _session_cache[key] = value
    logger.debug(f"Session updated: {key} = {value}")


def clear_session():
    """Clear all session data."""
    global _session_cache
    _session_cache = {}
    logger.info("Session cache cleared")


def reset_services():
    """
    Reset all service instances.
    Useful for testing or reinitializing.
    """
    global _db_manager, _file_manager, _training_service, _model_service, _hardware_service

    if _db_manager:
        _db_manager.close()

    _db_manager = None
    _file_manager = None
    _training_service = None
    _model_service = None
    _hardware_service = None

    # Also clear session cache on reset
    clear_session()

    logger.info("All services reset")
