"""
Device utility module for managing device selection and resolution.
Provides device-agnostic abstraction for CUDA, MPS, and CPU.
"""
import gc
import os
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
                "MPS is only available on macOS with Apple Silicon (M1/M2/M3/M4). "
                "Please use 'device: auto' to automatically select an available device, "
                "or 'device: cuda' for NVIDIA GPUs, or 'device: cpu' for CPU-only training."
            )
        device_obj = torch.device("mps")
        device_type = "mps"
        # Configure MPS memory management
        _configure_mps_memory()
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
            _configure_mps_memory()
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


def _configure_mps_memory() -> None:
    """
    Configure MPS memory management environment variables and settings.

    Sets watermark ratios to prevent MPS from over-allocating unified memory
    and crashing the system. Must be called before any MPS allocations.
    """
    # PYTORCH_MPS_HIGH_WATERMARK_RATIO: Hard limit for total allowed allocations.
    # Default is 1.7 (170% of recommendedMaxWorkingSetSize) which is too aggressive
    # for training workloads and can cause macOS to kill the process.
    # Setting to 0.0 disables the limit entirely, letting the OS manage memory.
    # This prevents PyTorch from throwing OOM errors prematurely while letting
    # macOS swap/compress memory naturally on unified memory systems.
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        logger.info("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (OS-managed memory limit)")

    # PYTORCH_MPS_LOW_WATERMARK_RATIO: Soft limit that triggers adaptive commit
    # (more frequent garbage collection and command buffer commits).
    # Only effective when HIGH_WATERMARK > 0. With HIGH_WATERMARK=0.0, this is unused.
    # But set it defensively in case the user overrides HIGH_WATERMARK.
    if "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
        logger.info("Set PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0")

    # PYTORCH_ENABLE_MPS_FALLBACK is set in app.py/cli.py before torch is imported.
    # Re-set here as a safety net (in case this module is used outside the app),
    # though it only takes effect if torch hasn't been imported yet.
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set PYTORCH_ENABLE_MPS_FALLBACK=1 (unsupported ops fall back to CPU)")

    # Set memory fraction to limit MPS allocations if the API is available
    try:
        if hasattr(torch.mps, "set_per_process_memory_fraction"):
            # 0.0 means unlimited (let OS handle), which is safest for training
            torch.mps.set_per_process_memory_fraction(0.0)
            logger.debug("Set MPS per-process memory fraction to 0.0 (unlimited/OS-managed)")
    except Exception as e:
        logger.debug(f"Could not set MPS memory fraction: {e}")


def get_device_memory_gb(device_type: str) -> float:
    """
    Get available memory for the specified device type in GB.

    Args:
        device_type: Device type string ("cuda", "mps", or "cpu")

    Returns:
        Available memory in GB (uses system RAM for MPS unified memory)
    """
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif device_type == "mps":
            # MPS uses unified memory - use system RAM as the relevant metric
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024 ** 3)
            logger.debug(f"MPS unified memory (system RAM): {total_gb:.2f}GB")
            return total_gb
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Could not get device memory: {e}")
        return 0.0


def get_mps_memory_info() -> dict:
    """
    Get detailed MPS memory usage information.

    Returns:
        Dictionary with current_allocated_mb, driver_allocated_mb, system_available_mb
    """
    info = {
        "current_allocated_mb": 0.0,
        "driver_allocated_mb": 0.0,
        "system_available_mb": 0.0,
        "system_total_mb": 0.0,
    }
    try:
        if hasattr(torch, "mps"):
            if hasattr(torch.mps, "current_allocated_memory"):
                info["current_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 ** 2)
            if hasattr(torch.mps, "driver_allocated_memory"):
                info["driver_allocated_mb"] = torch.mps.driver_allocated_memory() / (1024 ** 2)
        import psutil
        mem = psutil.virtual_memory()
        info["system_available_mb"] = mem.available / (1024 ** 2)
        info["system_total_mb"] = mem.total / (1024 ** 2)
    except Exception as e:
        logger.debug(f"Could not get MPS memory info: {e}")
    return info


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
    Clear device cache/memory for the specified device.

    For MPS, this performs garbage collection followed by MPS cache clearing
    to reclaim unified memory. For CUDA, clears the CUDA cache.

    Args:
        device_type: Device type string ("cuda", "mps", or "cpu")
    """
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        elif device_type == "mps":
            # Force Python garbage collection first to release Python-side references
            gc.collect()
            # Then clear the MPS allocator cache
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            logger.debug("Cleared MPS cache (gc.collect + torch.mps.empty_cache)")
    except Exception as e:
        logger.warning(f"Could not clear device cache: {e}")
