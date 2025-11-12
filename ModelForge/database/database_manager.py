"""
Database manager with SQLAlchemy and connection pooling.
Replaces the old DBManager with proper session management.
"""
import os
from typing import Optional, List, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from .models import Base, Model
from ..exceptions import DatabaseError
from ..logging_config import logger


class DatabaseManager:
    """
    Database manager using SQLAlchemy.
    Provides connection pooling and proper session management.
    """

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create engine with connection pooling
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,  # Set to True for SQL debugging
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Create tables if they don't exist
        self._initialize_database()

        logger.info(f"Database initialized at {db_path}")

    def _initialize_database(self):
        """Create tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}") from e

    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session using context manager.

        Yields:
            SQLAlchemy session

        Example:
            with db_manager.get_session() as session:
                models = session.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def add_model(
        self,
        model_id: str,
        name: str,
        base_model: str,
        task: str,
        path: str,
        strategy: str = "sft",
        provider: str = "huggingface",
        compute_profile: Optional[str] = None,
        config: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Add a model to the database.

        Args:
            model_id: Unique model identifier
            name: Model name
            base_model: Base model identifier
            task: Task type
            path: Path to model files
            strategy: Training strategy used
            provider: Model provider
            compute_profile: Compute profile used
            config: JSON configuration

        Returns:
            Dictionary of model data if successful, None otherwise
        """
        logger.info(f"Adding model to database: {model_id}")

        try:
            with self.get_session() as session:
                model = Model(
                    id=model_id,
                    name=name,
                    base_model=base_model,
                    task=task,
                    path=path,
                    strategy=strategy,
                    provider=provider,
                    compute_profile=compute_profile,
                    config=config,
                )
                session.add(model)
                session.commit()

                logger.info(f"Model added successfully: {model_id}")
                return model.to_dict()

        except Exception as e:
            logger.error(f"Error adding model: {e}")
            raise DatabaseError(f"Failed to add model: {str(e)}") from e

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary of model data if found, None otherwise
        """
        logger.info(f"Fetching model: {model_id}")

        try:
            with self.get_session() as session:
                model = session.query(Model).filter_by(id=model_id).first()
                if model:
                    return model.to_dict()
                else:
                    logger.warning(f"Model not found: {model_id}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching model: {e}")
            raise DatabaseError(f"Failed to fetch model: {str(e)}") from e

    def get_all_models(self) -> List[Dict]:
        """
        Get all models from the database.

        Returns:
            List of model dictionaries
        """
        logger.info("Fetching all models")

        try:
            with self.get_session() as session:
                models = session.query(Model).filter_by(is_active=True).all()
                return [model.to_dict() for model in models]

        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            raise DatabaseError(f"Failed to fetch models: {str(e)}") from e

    def update_model(self, model_id: str, **kwargs) -> Optional[Dict]:
        """
        Update a model in the database.

        Args:
            model_id: Model identifier
            **kwargs: Fields to update

        Returns:
            Updated model dictionary if successful, None otherwise
        """
        logger.info(f"Updating model: {model_id}")

        try:
            with self.get_session() as session:
                model = session.query(Model).filter_by(id=model_id).first()
                if not model:
                    logger.warning(f"Model not found for update: {model_id}")
                    return None

                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)

                session.commit()
                logger.info(f"Model updated successfully: {model_id}")
                return model.to_dict()

        except Exception as e:
            logger.error(f"Error updating model: {e}")
            raise DatabaseError(f"Failed to update model: {str(e)}") from e

    def delete_model(self, model_id: str) -> bool:
        """
        Soft delete a model (mark as inactive).

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deleting model: {model_id}")

        try:
            with self.get_session() as session:
                model = session.query(Model).filter_by(id=model_id).first()
                if not model:
                    logger.warning(f"Model not found for deletion: {model_id}")
                    return False

                model.is_active = False
                session.commit()
                logger.info(f"Model deleted successfully: {model_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            raise DatabaseError(f"Failed to delete model: {str(e)}") from e

    def get_models_by_task(self, task: str) -> List[Dict]:
        """
        Get all models for a specific task.

        Args:
            task: Task type

        Returns:
            List of model dictionaries
        """
        logger.info(f"Fetching models for task: {task}")

        try:
            with self.get_session() as session:
                models = session.query(Model).filter_by(
                    task=task,
                    is_active=True
                ).all()
                return [model.to_dict() for model in models]

        except Exception as e:
            logger.error(f"Error fetching models by task: {e}")
            raise DatabaseError(f"Failed to fetch models by task: {str(e)}") from e

    def close(self):
        """Close the database engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
