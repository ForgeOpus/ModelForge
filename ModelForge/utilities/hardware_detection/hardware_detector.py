import psutil
import pynvml
import torch
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

            # Fields expected by HardwareService
            self.gpu_count = 0
            self.gpu_name = None
            # GPU memory (bytes)
            self.total_memory = 0
            self.available_memory = 0
            # Driver / CUDA details
            self.driver_version = None
            self.cuda_version = None
            # Classified compute profile
            self.compute_profile = None
            
            logging.info("HardwareDetector initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize HardwareDetector: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_gpu_specs(self) -> None:
        """
        Get GPU specifications with enhanced error handling.
        Supports both NVIDIA CUDA and Apple MPS devices.
        
        Raises:
            RuntimeError: If GPU detection fails
        """
        # First, try to detect NVIDIA CUDA GPU
        cuda_available = False
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                cuda_available = True
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)

                # Populate fields expected by HardwareService
                self.gpu_count = device_count
                self.gpu_name = gpu_name
                self.total_memory = gpu_mem_info.total  # bytes
                self.available_memory = getattr(gpu_mem_info, 'free', 0)  # bytes

                # Driver and CUDA versions (best effort)
                try:
                    drv_ver = pynvml.nvmlSystemGetDriverVersion()
                    # nvml may return bytes or str depending on binding
                    self.driver_version = drv_ver.decode('utf-8') if hasattr(drv_ver, 'decode') else str(drv_ver)
                except Exception:
                    self.driver_version = None
                try:
                    # nvmlSystemGetCudaDriverVersion_v2 returns int like major*1000 + minor*10
                    if hasattr(pynvml, 'nvmlSystemGetCudaDriverVersion_v2'):
                        cuda_int = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                        major = cuda_int // 1000
                        minor = (cuda_int % 1000) // 10
                        self.cuda_version = f"{major}.{minor}"
                    else:
                        self.cuda_version = None
                except Exception:
                    self.cuda_version = None

                # Also keep existing hardware_profile entries in GB for prior consumers
                # Guard against unexpected string types from bindings
                try:
                    numeric_total = float(self.total_memory)
                except (TypeError, ValueError):
                    numeric_total = 0.0
                gpu_total_mem_gb = numeric_total / (1024 ** 3) if numeric_total else 0.0
                self.hardware_profile['gpu_name'] = gpu_name
                self.hardware_profile['gpu_total_memory_gb'] = round(gpu_total_mem_gb, 2)
                self.hardware_profile['device_type'] = 'cuda'
                
                logging.info(f"CUDA GPU detected: {gpu_name} with {gpu_total_mem_gb:.2f}GB memory; GPUs: {device_count}")
            
        except Exception as e:
            logging.debug(f"CUDA GPU detection failed (this is normal on non-NVIDIA systems): {str(e)}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass  # Ignore shutdown errors
        
        # If no CUDA GPU, try to detect Apple MPS
        if not cuda_available:
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS is available
                    memory = psutil.virtual_memory()
                    unified_memory_gb = memory.total / (1024 ** 3)
                    
                    self.gpu_count = 1
                    self.gpu_name = "Apple M-series (MPS)"
                    # MPS uses unified memory - use system RAM as approximation
                    self.total_memory = memory.total
                    self.available_memory = memory.available
                    self.driver_version = None
                    self.cuda_version = None
                    
                    self.hardware_profile['gpu_name'] = "Apple M-series (MPS)"
                    self.hardware_profile['gpu_total_memory_gb'] = round(unified_memory_gb, 2)
                    self.hardware_profile['device_type'] = 'mps'
                    
                    logging.info(f"Apple MPS detected with {unified_memory_gb:.2f}GB unified memory")
                    return
            except Exception as e:
                logging.debug(f"MPS detection failed: {str(e)}")
        
        # If neither CUDA nor MPS available, raise error
        if not cuda_available and not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError(
                "No GPU detected. ModelForge requires either an NVIDIA GPU with CUDA "
                "or Apple Silicon with MPS support. If you have a GPU, please ensure "
                "you have the correct drivers installed."
            )

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
            # Expose for consumers like HardwareService
            self.compute_profile = profile
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
            "mps_available": False,
            "device_type": None,
            "error": None
        }
        
        try:
            # Try to get basic GPU info (NVIDIA CUDA)
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                system_info["gpu_available"] = True
                system_info["cuda_available"] = True
                system_info["device_type"] = "cuda"
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                system_info["gpu_name"] = pynvml.nvmlDeviceGetName(gpu_handle)
            pynvml.nvmlShutdown()
        except Exception as e:
            system_info["error"] = f"CUDA GPU detection failed: {str(e)}"
        
        # Try to detect MPS if CUDA not available
        if not system_info["cuda_available"]:
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    system_info["gpu_available"] = True
                    system_info["mps_available"] = True
                    system_info["device_type"] = "mps"
                    system_info["gpu_name"] = "Apple M-series (MPS)"
            except Exception as e:
                if system_info["error"]:
                    system_info["error"] += f"; MPS detection failed: {str(e)}"
                else:
                    system_info["error"] = f"MPS detection failed: {str(e)}"
        
        try:
            # Get basic system info
            memory = psutil.virtual_memory()
            system_info["ram_total_gb"] = round(memory.total / (1024 ** 3), 0)
            system_info["cpu_cores"] = psutil.cpu_count(logical=True)
        except Exception as e:
            system_info["error"] = f"System info detection failed: {str(e)}"
        
        return system_info