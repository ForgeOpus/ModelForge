"""
SQLAlchemy models for ModelForge database.
"""
from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Model(Base):
    """Model for storing fine-tuned model metadata."""

    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    task = Column(String, nullable=False)
    strategy = Column(String, default="sft")
    provider = Column(String, default="huggingface")
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    compute_profile = Column(String, nullable=True)
    config = Column(Text, nullable=True)  # JSON config
    is_active = Column(Boolean, default=True)

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "base_model": self.base_model,
            "task": self.task,
            "strategy": self.strategy,
            "provider": self.provider,
            "path": self.path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "compute_profile": self.compute_profile,
            "config": self.config,
            "is_active": self.is_active,
        }
