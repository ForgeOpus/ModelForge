"""Select trainer precision from the devices that will execute training."""
import torch


def training_precision(hardware):
    cuda = [gpu for gpu in hardware.gpus if isinstance(gpu["total_memory_gb"], (int, float))]
    if not cuda or not torch.cuda.is_available():
        return False, False
    try:
        bf16 = all(torch.cuda.get_device_capability(index)[0] >= 8 for index in range(len(cuda)))
    except (RuntimeError, AssertionError):
        bf16 = False
    return bf16, not bf16
