"""
Hardware service for detecting and managing hardware capabilities.
Wraps hardware detection functionality.
"""
from typing import Dict, List, Optional

from ..utilities.hardware_detection.hardware_detector import HardwareDetector
from ..utilities.hardware_detection.model_recommendation import ModelRecommendation
from ..logging_config import logger


class HardwareService:
    """Service for hardware detection and model recommendations."""

    def __init__(self):
        """Initialize hardware service."""
        self.hardware_detector = HardwareDetector()
        self.model_recommendation = ModelRecommendation()
        logger.info("Hardware service initialized")

    def get_hardware_specs(self) -> Dict:
        """
        Get hardware specifications.

        Returns:
            Dictionary with hardware specifications
        """
        logger.info("Detecting hardware specifications")

        specs = {
            "gpu_count": self.hardware_detector.gpu_count,
            "gpu_name": self.hardware_detector.gpu_name,
            "total_memory_gb": self.hardware_detector.total_memory / (1024 ** 3),
            "available_memory_gb": self.hardware_detector.available_memory / (1024 ** 3),
            "driver_version": self.hardware_detector.driver_version,
            "cuda_version": self.hardware_detector.cuda_version,
            "compute_profile": self.hardware_detector.compute_profile,
        }

        logger.info(f"Hardware specs: {specs}")
        return specs

    def get_compute_profile(self) -> str:
        """
        Get compute profile (low_end, mid_range, high_end).

        Returns:
            Compute profile string
        """
        return self.hardware_detector.compute_profile

    def get_recommended_models(self, task: str) -> Dict:
        """
        Get recommended models for a task based on hardware.

        Args:
            task: Task type

        Returns:
            Dictionary with recommended models
        """
        logger.info(f"Getting model recommendations for task: {task}")

        compute_profile = self.get_compute_profile()
        recommendations = self.model_recommendation.get_model_recommendations(
            task=task,
            compute_profile=compute_profile
        )

        return {
            "compute_profile": compute_profile,
            "task": task,
            "recommendations": recommendations,
        }

    def validate_batch_size(self, batch_size: int, compute_profile: str) -> bool:
        """
        Validate if batch size is appropriate for compute profile.

        Args:
            batch_size: Batch size to validate
            compute_profile: Compute profile

        Returns:
            True if valid, False otherwise
        """
        # High-end can handle any batch size
        if compute_profile == "high_end":
            return True

        # Mid-range and low-end should use smaller batch sizes
        if compute_profile == "mid_range" and batch_size <= 4:
            return True

        if compute_profile == "low_end" and batch_size <= 2:
            return True

        return False
