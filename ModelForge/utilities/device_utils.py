"""
Device utility module for managing device selection and resolution.
Provides device-agnostic abstraction for CUDA, MPS, and CPU.
"""
import torch
from typing import Tuple, List
from ..logging_config import logger
from ..exceptions import ConfigurationError


def resolve_device(device: str) -> Tuple[torch.device, str]:
    """
    Resolve device string to torch.device and device type string.
    
    Args:
        device: Device selection ("auto", "cuda", "mps", or "cpu")
        
    Returns:
        Tuple of (torch.device, device_type_string)
        
    Raises:
        ConfigurationError: If requested device is not available
        
    Examples:
        >>> device_obj, device_type = resolve_device("auto")
        >>> # Returns: (torch.device("cuda:0"), "cuda") if CUDA available
        
        >>> device_obj, device_type = resolve_device("mps")
        >>> # Returns: (torch.device("mps"), "mps") if MPS available
    """
    logger.info(f"Resolving device: {device}")
    
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError(
                "CUDA device requested but CUDA is not available. "
                "Please ensure you have an NVIDIA GPU and CUDA installed, "
                "or use 'device: auto' to automatically select an available device."
            )
        device_obj = torch.device("cuda:0")
        device_type = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
    elif device == "mps":
        # Check if MPS is available (macOS with Apple Silicon)
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ConfigurationError(
                "MPS device requested but MPS is not available. "
                "MPS is only available on macOS with Apple Silicon (M1/M2/M3). "
                "Please use 'device: auto' to automatically select an available device, "
                "or 'device: cuda' for NVIDIA GPUs, or 'device: cpu' for CPU-only training."
            )
        device_obj = torch.device("mps")
        device_type = "mps"
        logger.info("Using Apple MPS device")
        
    elif device == "cpu":
        device_obj = torch.device("cpu")
        device_type = "cpu"
        logger.info("Using CPU device (training will be slow)")
        
    elif device == "auto":
        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device_obj = torch.device("cuda:0")
            device_type = "cuda"
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
            device_type = "mps"
            logger.info("Auto-detected Apple MPS device")
        else:
            device_obj = torch.device("cpu")
            device_type = "cpu"
            logger.info("No GPU detected, using CPU device (training will be slow)")
    else:
        raise ConfigurationError(
            f"Invalid device: {device}. Must be one of: 'auto', 'cuda', 'mps', 'cpu'"
        )
    
    return device_obj, device_type


def get_device_memory_gb(device_type: str) -> float:
    """
    Get available memory for the specified device type in GB.
    
    Args:
        device_type: Device type string ("cuda", "mps", or "cpu")
        
    Returns:
        Available memory in GB (0 if not available or unknown)
    """
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            # Get CUDA device memory
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif device_type == "mps":
            # MPS doesn't have a direct memory query API
            # Return 0 to indicate unknown (caller should use system RAM as approximation)
            return 0.0
        else:
            # CPU - not applicable
            return 0.0
    except Exception as e:
        logger.warning(f"Could not get device memory: {e}")
        return 0.0


def is_device_available(device_type: str) -> bool:
    """
    Check if a specific device type is available.
    
    Args:
        device_type: Device type to check ("cuda", "mps", or "cpu")
        
    Returns:
        True if device is available, False otherwise
    """
    if device_type == "cuda":
        return torch.cuda.is_available()
    elif device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    elif device_type == "cpu":
        return True
    else:
        return False


def get_available_devices() -> List[str]:
    """
    Get list of all available device types on this system.
    
    Returns:
        List of available device type strings
    """
    devices = ["cpu"]  # CPU always available
    
    if torch.cuda.is_available():
        devices.append("cuda")
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
        
    return devices


def clear_device_cache(device_type: str) -> None:
    """
    Clear device cache/memory if supported by the device.
    
    Args:
        device_type: Device type string ("cuda", "mps", or "cpu")
    """
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        # MPS and CPU don't have explicit cache clearing
    except Exception as e:
        logger.warning(f"Could not clear device cache: {e}")
