'''
Hardware detection utilities to determine system resources like:
- RAM
- VRAM
- CPU cores
- GPU availability and type
- Disk space
This information can be used to optimize model loading, fine-tuning strategies, 
and resource management in ModelForge.
'''

from pydantic import BaseModel, Field, field_validator
from pynvml import *
import psutil
import torch

class HardwareDetector:

    class HardwareSpecs(BaseModel):
        physical_core_count: int = Field(..., description="Number of physical CPU cores")
        disk_storage_gb: float = Field(..., description="Total disk storage in GB")
        available_disk_storage_gb: float = Field(..., description="Available disk storage in GB")
        ram_gb: float = Field(..., description="Total RAM in GB")
        gpus: list = Field(..., description="List of GPU details including name and VRAM in GB")

        def __str__(self):
            gpu_info_str = "\n".join([f"  - {gpu['name']} with {gpu['total_memory_gb'] if isinstance(gpu['total_memory_gb'], (int, float)) else 'N/A'} GB VRAM" for gpu in self.gpus])
            return (f"Hardware Specifications:\n"
                    f"- Physical CPU Cores: {self.physical_core_count}\n"
                    f"- Total Disk Storage: {self.disk_storage_gb:.2f} GB\n"
                    f"- Available Disk Storage: {self.available_disk_storage_gb:.2f} GB\n"
                    f"- Total RAM: {self.ram_gb:.2f} GB\n"
                    f"- GPUs:\n{gpu_info_str if gpu_info_str else '  None'}")

        @field_validator('gpus', mode='before')
        def validate_gpus(cls, v):
            if not isinstance(v, list):
                raise ValueError("GPUs must be a list of GPU details")
            for gpu in v:
                if 'name' not in gpu or 'total_memory_gb' not in gpu:
                    raise ValueError("Each GPU entry must contain 'name' and 'total_memory_gb'")
                if isinstance(gpu['total_memory_gb'], (int, float)) and gpu['total_memory_gb'] < 0:
                    raise ValueError("GPU memory must be non-negative")
            return v

        @field_validator('physical_core_count', 'disk_storage_gb', 'available_disk_storage_gb', 'ram_gb')
        def validate_positive(cls, v):
            if v < 0:
                raise ValueError("Hardware specifications must be non-negative")
            return v

        @field_validator('disk_storage_gb', 'available_disk_storage_gb', 'ram_gb')
        def validate_numeric(cls, v):
            if not isinstance(v, (int, float)):
                raise ValueError("Disk storage and RAM must be numeric values")
            return v

        @field_validator('physical_core_count')
        def validate_core_count(cls, v):
            if not isinstance(v, int):
                raise ValueError("Physical core count must be an integer")
            return v

    @staticmethod
    def get_gpu_info():
        try:
            nvmlInit()
            gpu_count = nvmlDeviceGetCount()
            gpu_info = []
            for i in range(gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                # Convert to GB
                total_memory = nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)
                
                gpu_info.append({
                    'name': name,
                    'total_memory_gb': total_memory
                })
            return gpu_info
        except Exception:
            # If NVML initialization or device query fails, return empty list
            return []
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass

    @classmethod
    def get_device_info_non_mps(cls):
        
        physical_core_count = psutil.cpu_count(logical=False)
        disk_storage = psutil.disk_usage('/').total / (1024 ** 3)  # Convert to GB
        available_disk_storage = psutil.disk_usage('/').free / (1024 ** 3)  # Convert to GB
        ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
        gpu_details = cls.get_gpu_info()

        return {
            'physical_core_count': physical_core_count,
            'disk_storage_gb': disk_storage,
            'available_disk_storage_gb': available_disk_storage,
            'ram_gb': ram,
            'gpus': gpu_details 
        }

    @staticmethod
    def get_device_info_mps():
        physical_core_count = psutil.cpu_count(logical=False)
        disk_storage = psutil.disk_usage('/').total / (1024 ** 3)  # Convert to GB
        available_disk_storage = psutil.disk_usage('/').free / (1024 ** 3)  # Convert to GB
        ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB

        gpu_details = []
        if torch.backends.mps.is_available():
            gpu_details.append({
                'name': 'Apple Silicon GPU',
                'total_memory_gb': 'N/A'  # VRAM is not directly accessible on MPS
            })

        return {
            'physical_core_count': physical_core_count,
            'disk_storage_gb': disk_storage,
            'available_disk_storage_gb': available_disk_storage,
            'ram_gb': ram,
            'gpus': gpu_details 
        }

    @classmethod
    def detect_hardware(cls):
        specs = cls.get_device_info_non_mps() if not torch.backends.mps.is_available() else cls.get_device_info_mps()
        response = cls.HardwareSpecs(**specs)
        return response