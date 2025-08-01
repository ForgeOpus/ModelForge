import psutil
import pynvml
import logging
from typing import Dict, List, Tuple, Union
from .config_manager import ConfigurationManager
from .model_recommendation import ModelRecommendationEngine


class HardwareDetector:
    def __init__(self):
        """
        Initialize HardwareDetector with enhanced error handling.
        
        Raises:
            RuntimeError: If critical initialization fails
        """
        try:
            # Initialize configuration manager
            self.config_manager = ConfigurationManager()
            
            # Initialize model recommendation engine
            self.model_recommendation_engine = ModelRecommendationEngine(self.config_manager)
            
            # Hardware detection attributes
            self.hardware_profile = {}
            self.model_requirements = {}
            self.model_recommendation = ""
            
            logging.info("HardwareDetector initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize HardwareDetector: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_gpu_specs(self) -> None:
        """
        Get GPU specifications with enhanced error handling.
        
        Raises:
            RuntimeError: If GPU detection fails
        """
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                raise RuntimeError("No CUDA-enabled GPU detected. Please ensure that your system has a CUDA-enabled GPU and that you have the correct drivers installed.")
            
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
            gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_total_mem = gpu_mem_info.total / (1024 ** 3)
            
            self.hardware_profile['gpu_name'] = gpu_name
            self.hardware_profile['gpu_total_memory_gb'] = round(gpu_total_mem, 2)
            
            logging.info(f"GPU detected: {gpu_name} with {gpu_total_mem:.2f}GB memory")
            
        except Exception as e:
            error_msg = f"GPU detection failed: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass  # Ignore shutdown errors

    def get_computer_specs(self) -> None:
        """
        Get computer specifications with enhanced error handling.
        
        Raises:
            RuntimeError: If system specs detection fails
        """
        try:
            memory = psutil.virtual_memory()
            ram_total = memory.total
            available_diskspace = psutil.disk_usage('/').free / (1024 ** 3)
            cpu_cores = psutil.cpu_count(logical=True)
            
            self.hardware_profile['ram_total_gb'] = round(ram_total / (1024 ** 3), 0)
            self.hardware_profile['available_diskspace_gb'] = round(available_diskspace, 2)
            self.hardware_profile['cpu_cores'] = cpu_cores
            
            logging.info(f"System specs: {self.hardware_profile['ram_total_gb']}GB RAM, {cpu_cores} CPU cores, {available_diskspace:.2f}GB disk space")
            
        except Exception as e:
            error_msg = f"System specs detection failed: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def classify_hardware_profile(self) -> str:
        """
        Classify hardware into performance profile using configuration thresholds.
        
        Returns:
            Hardware profile string
            
        Raises:
            RuntimeError: If classification fails
        """
        try:
            if not self.hardware_profile:
                raise ValueError("Hardware profile is empty. Run hardware detection first.")
            
            gpu_memory_thresholds = self.config_manager.get_gpu_memory_thresholds()
            ram_thresholds = self.config_manager.get_ram_thresholds()
            
            gpu_memory_gb = self.hardware_profile.get('gpu_total_memory_gb', 0)
            ram_gb = self.hardware_profile.get('ram_total_gb', 0)
            
            # Use configuration thresholds with fallback defaults
            low_end_gpu_threshold = gpu_memory_thresholds.get("low_end_max", 7.2)
            mid_range_gpu_threshold = gpu_memory_thresholds.get("mid_range_max", 15.2)
            low_end_ram_threshold = ram_thresholds.get("low_end_max", 15.2)
            
            # Apply classification logic
            if gpu_memory_gb < low_end_gpu_threshold:
                profile = 'low_end'
            elif gpu_memory_gb < mid_range_gpu_threshold:
                profile = 'low_end' if ram_gb < low_end_ram_threshold else 'mid_range'
            else:
                profile = 'mid_range' if ram_gb < low_end_ram_threshold else 'high_end'
            
            logging.info(f"Hardware classified as: {profile}")
            return profile
            
        except Exception as e:
            error_msg = f"Hardware classification failed: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def run(self, task: str) -> Tuple[Dict[str, Union[str, float]], Dict[str, Union[str, float]], str, List[str]]:
        """
        Run hardware detection and model recommendation with enhanced error handling.
        
        Args:
            task: Task name for model recommendation
            
        Returns:
            Tuple of (model_requirements, hardware_profile, model_recommendation, alternatives)
            
        Raises:
            ValueError: If task is not supported
            RuntimeError: If detection or recommendation fails
        """
        try:
            # Validate task is supported
            if not self.config_manager.is_task_supported(task):
                supported_tasks = self.config_manager.get_task_names()
                raise ValueError(f"Unsupported task '{task}'. Supported tasks: {supported_tasks}")
            
            logging.info(f"Running hardware detection for task: {task}")
            
            # Set task
            self.model_requirements['task'] = task
            
            # Detect hardware specs
            self.get_computer_specs()
            self.get_gpu_specs()
            
            # Classify hardware profile
            profile = self.classify_hardware_profile()
            self.model_requirements['profile'] = profile
            
            # Get model recommendation
            primary_model, alternative_models = self.model_recommendation_engine.get_recommendation(profile, task)
            self.model_recommendation = primary_model
            
            logging.info(f"Recommended model: {primary_model}")
            logging.info(f"Alternative models: {alternative_models}")
            
            return self.model_requirements, self.hardware_profile, self.model_recommendation, alternative_models
            
        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            error_msg = f"Hardware detection run failed: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_system_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get basic system information without full hardware detection.
        Useful for troubleshooting and system validation.
        
        Returns:
            Dictionary with basic system information
        """
        system_info = {
            "gpu_available": False,
            "cuda_available": False,
            "error": None
        }
        
        try:
            # Try to get basic GPU info
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                system_info["gpu_available"] = True
                system_info["cuda_available"] = True
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                system_info["gpu_name"] = pynvml.nvmlDeviceGetName(gpu_handle)
            pynvml.nvmlShutdown()
        except Exception as e:
            system_info["error"] = f"GPU detection failed: {str(e)}"
        
        try:
            # Get basic system info
            memory = psutil.virtual_memory()
            system_info["ram_total_gb"] = round(memory.total / (1024 ** 3), 0)
            system_info["cpu_cores"] = psutil.cpu_count(logical=True)
        except Exception as e:
            system_info["error"] = f"System info detection failed: {str(e)}"
        
        return system_info