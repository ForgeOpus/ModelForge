"""
Hardware service for detecting and managing hardware capabilities.
Wraps hardware detection functionality.
"""
from typing import Dict, List, Optional

from ..utilities.hardware_detection.hardware_detector import HardwareDetector
from ..utilities.hardware_detection.model_recommendation import ModelRecommendationEngine
from ..logging_config import logger


class HardwareService:
    """Service for hardware detection and model recommendations."""

    def __init__(self):
        """Initialize hardware service."""
        self.hardware_detector = HardwareDetector()
        self.model_recommendation = ModelRecommendationEngine()
        self._detected = False  # Track if hardware detection has run
        logger.info("Hardware service initialized")

    def _ensure_detected(self):
        """
        Ensure hardware detection has been performed.
        Runs detection on first call, then caches results.
        """
        if not self._detected:
            logger.info("Running hardware detection...")
            try:
                # Run detection sequence
                self.hardware_detector.get_computer_specs()
                self.hardware_detector.get_gpu_specs()
                self.hardware_detector.classify_hardware_profile()
                self._detected = True
                logger.info(f"Hardware detection complete. Profile: {self.hardware_detector.compute_profile}")
            except Exception as e:
                logger.error(f"Hardware detection failed: {e}")
                raise

    def get_hardware_specs(self) -> Dict:
        """
        Get hardware specifications.

        Returns:
            Dictionary with hardware specifications
        """
        # Ensure hardware detection has run
        self._ensure_detected()

        logger.info("Getting hardware specifications")

        # Get hardware profile data
        hw_profile = self.hardware_detector.hardware_profile

        specs = {
            "gpu_count": self.hardware_detector.gpu_count,
            "gpu_name": self.hardware_detector.gpu_name or hw_profile.get("gpu_name", "Unknown"),
            "gpu_memory_gb": hw_profile.get("gpu_total_memory_gb", 0),
            "ram_gb": hw_profile.get("ram_total_gb", 0),
            "disk_space_gb": hw_profile.get("available_diskspace_gb", 0),
            "cpu_cores": hw_profile.get("cpu_cores", 0),
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
        # Ensure hardware detection has run
        self._ensure_detected()
        return self.hardware_detector.compute_profile

    def get_recommended_models(self, task: str) -> Dict:
        """
        Get recommended models for a task based on hardware.

        Args:
            task: Task type

        Returns:
            Dictionary with recommended models
        """
        # Ensure hardware detection has run
        self._ensure_detected()

        logger.info(f"Getting model recommendations for task: {task}")

        compute_profile = self.get_compute_profile()

        # Get recommendation returns a tuple of (primary_model, alternative_models)
        primary_model, alternative_models = self.model_recommendation.get_recommendation(
            hardware_profile=compute_profile,
            task=task
        )

        return {
            "compute_profile": compute_profile,
            "task": task,
            "recommended_model": primary_model,
            "possible_models": alternative_models,
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
