import logging
from typing import Optional, Dict, Any
from .hardware_detector import HardwareDetector
from .config_manager import ConfigurationManager


class HardwareDetectorFactory:
    """
    Factory for creating HardwareDetector instances with proper error handling
    and configuration management.
    """
    
    @staticmethod
    def create() -> HardwareDetector:
        """
        Create a HardwareDetector instance with enhanced error handling.
        
        Returns:
            HardwareDetector instance
            
        Raises:
            RuntimeError: If detector creation fails
        """
        try:
            return HardwareDetector()
        except Exception as e:
            error_msg = f"Failed to create HardwareDetector: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def create_with_fallback() -> Optional[HardwareDetector]:
        """
        Create a HardwareDetector instance with fallback handling.
        Returns None if creation fails instead of raising an exception.
        
        Returns:
            HardwareDetector instance or None if creation fails
        """
        try:
            return HardwareDetectorFactory.create()
        except Exception as e:
            logging.warning(f"HardwareDetector creation failed, returning None: {str(e)}")
            return None
    
    @staticmethod
    def validate_system_requirements() -> Dict[str, Any]:
        """
        Validate system requirements before creating detector.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "gpu_available": False,
            "cuda_available": False,
            "config_files_valid": False,
            "errors": [],
            "warnings": []
        }
        
        # Check GPU availability
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                validation_results["gpu_available"] = True
                validation_results["cuda_available"] = True
            else:
                validation_results["warnings"].append("No CUDA-enabled GPU detected")
            pynvml.nvmlShutdown()
        except Exception as e:
            validation_results["errors"].append(f"GPU detection failed: {str(e)}")
        
        # Check configuration files
        try:
            config_manager = ConfigurationManager()
            # Try to load basic configurations
            config_manager.get_model_profiles()
            config_manager.get_hardware_thresholds()
            config_manager.get_available_tasks()
            validation_results["config_files_valid"] = True
        except Exception as e:
            validation_results["errors"].append(f"Configuration validation failed: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def create_with_validation() -> HardwareDetector:
        """
        Create HardwareDetector with pre-creation validation.
        
        Returns:
            HardwareDetector instance
            
        Raises:
            RuntimeError: If validation fails or detector creation fails
        """
        # Validate system requirements
        validation = HardwareDetectorFactory.validate_system_requirements()
        
        if validation["errors"]:
            error_msg = f"System validation failed: {'; '.join(validation['errors'])}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logging.warning(warning)
        
        # Create detector if validation passes
        return HardwareDetectorFactory.create() 