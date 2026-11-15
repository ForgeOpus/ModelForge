import pytest
from utilities.hardware_detector import HardwareDetector

def test_hardware_spec_response_validation():
    # Test that the response from detect_hardware can be validated against the HardwareSpecs model
    response = HardwareDetector.detect_hardware()

    print(f"Recieved hardware specs: \n{response}\n")

    assert isinstance(response, HardwareDetector.HardwareSpecs)
    assert response.physical_core_count > 0
    assert response.disk_storage_gb > 0
    assert response.available_disk_storage_gb >= 0
    assert response.ram_gb > 0
    assert isinstance(response.gpus, list)

def test_negative_storage():
    # Test that providing negative storage raises a ValidationError
    with pytest.raises(Exception) as excinfo:
        HardwareDetector.HardwareSpecs(
            physical_core_count=4,
            disk_storage_gb=-500,  # Invalid negative storage
            available_disk_storage_gb=100,
            ram_gb=16,
            gpus=[]
        )
    assert "must be non-negative" in str(excinfo.value)

def test_negative_ram():
    # Test that providing negative RAM raises a ValidationError
    with pytest.raises(Exception) as excinfo:
        HardwareDetector.HardwareSpecs(
            physical_core_count=4,
            disk_storage_gb=500,
            available_disk_storage_gb=100,
            ram_gb=-16,  # Invalid negative RAM
            gpus=[]
        )
    assert "must be non-negative" in str(excinfo.value)

def test_negative_gpu_memory():
    # Test that providing negative GPU memory raises a ValidationError
    with pytest.raises(Exception) as excinfo:
        HardwareDetector.HardwareSpecs(
            physical_core_count=4,
            disk_storage_gb=500,
            available_disk_storage_gb=100,
            ram_gb=16,
            gpus=[{'name': 'Test GPU', 'total_memory_gb': -8}]  # Invalid negative GPU memory
        )
    assert "GPU memory must be non-negative" in str(excinfo.value)

def test_gpu_info_structure():
    # Test that the GPU information in the response has the expected structure
    response = HardwareDetector.detect_hardware()
    print(f"Recieved hardware specs for GPU structure test: \n{response}\n")
    for gpu in response.gpus:
        assert 'name' in gpu, "GPU info should contain 'name'"
        assert 'total_memory_gb' in gpu, "GPU info should contain 'total_memory_gb'"

# ---------------------------------------------------------------------------
# HardwareSpecs.__str__
# ---------------------------------------------------------------------------

def test_hardware_specs_str_contains_core_count():
    specs = HardwareDetector.HardwareSpecs(
        physical_core_count=8,
        disk_storage_gb=500.0,
        available_disk_storage_gb=200.0,
        ram_gb=32.0,
        gpus=[]
    )
    assert "8" in str(specs)

def test_hardware_specs_str_with_no_gpus_shows_none():
    specs = HardwareDetector.HardwareSpecs(
        physical_core_count=4,
        disk_storage_gb=500.0,
        available_disk_storage_gb=200.0,
        ram_gb=16.0,
        gpus=[]
    )
    assert "None" in str(specs)

def test_hardware_specs_str_with_gpu_shows_name_and_vram():
    specs = HardwareDetector.HardwareSpecs(
        physical_core_count=4,
        disk_storage_gb=500.0,
        available_disk_storage_gb=200.0,
        ram_gb=16.0,
        gpus=[{'name': 'RTX 4090', 'total_memory_gb': 24.0}]
    )
    output = str(specs)
    assert "RTX 4090" in output
    assert "24" in output

def test_hardware_specs_non_int_core_count_raises():
    with pytest.raises(Exception):
        HardwareDetector.HardwareSpecs(
            physical_core_count="four",
            disk_storage_gb=500.0,
            available_disk_storage_gb=200.0,
            ram_gb=16.0,
            gpus=[]
        )

# ---------------------------------------------------------------------------
# get_gpu_info
# ---------------------------------------------------------------------------

def test_get_gpu_info_returns_list():
    result = HardwareDetector.get_gpu_info()
    assert isinstance(result, list)

def test_get_gpu_info_nvml_failure_returns_empty():
    from unittest.mock import patch
    with patch("utilities.hardware_detector.nvmlInit", side_effect=Exception("NVML failed")):
        result = HardwareDetector.get_gpu_info()
    assert result == []

# ---------------------------------------------------------------------------
# get_device_info_non_mps
# ---------------------------------------------------------------------------

def test_get_device_info_non_mps_returns_expected_keys():
    from unittest.mock import patch, MagicMock
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 4
    mock_psutil.disk_usage.return_value = MagicMock(total=500 * (1024 ** 3), free=200 * (1024 ** 3))
    mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024 ** 3))
    with patch("utilities.hardware_detector.psutil", mock_psutil):
        with patch.object(HardwareDetector, "get_gpu_info", return_value=[]):
            result = HardwareDetector.get_device_info_non_mps()
    assert set(result.keys()) == {"physical_core_count", "disk_storage_gb", "available_disk_storage_gb", "ram_gb", "gpus"}

# ---------------------------------------------------------------------------
# get_device_info_mps
# ---------------------------------------------------------------------------

def test_get_device_info_mps_no_gpu_when_unavailable():
    from unittest.mock import patch, MagicMock
    import torch
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.disk_usage.return_value = MagicMock(total=500 * (1024 ** 3), free=200 * (1024 ** 3))
    mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024 ** 3))
    with patch("utilities.hardware_detector.psutil", mock_psutil):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            result = HardwareDetector.get_device_info_mps()
    assert result["gpus"] == []

def test_get_device_info_mps_adds_apple_silicon_gpu():
    from unittest.mock import patch, MagicMock
    import torch
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.disk_usage.return_value = MagicMock(total=500 * (1024 ** 3), free=200 * (1024 ** 3))
    mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024 ** 3))
    with patch("utilities.hardware_detector.psutil", mock_psutil):
        with patch.object(torch.backends.mps, "is_available", return_value=True):
            result = HardwareDetector.get_device_info_mps()
    assert len(result["gpus"]) == 1
    assert result["gpus"][0]["name"] == "Apple Silicon GPU"

# ---------------------------------------------------------------------------
# detect_hardware
# ---------------------------------------------------------------------------

def test_detect_hardware_uses_mps_branch_on_apple():
    from unittest.mock import patch
    import torch
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        with patch.object(HardwareDetector, "get_device_info_mps", return_value={
            "physical_core_count": 8,
            "disk_storage_gb": 500.0,
            "available_disk_storage_gb": 200.0,
            "ram_gb": 16.0,
            "gpus": [{"name": "Apple Silicon GPU", "total_memory_gb": "N/A"}]
        }) as mock_mps:
            with patch.object(HardwareDetector, "get_device_info_non_mps") as mock_non_mps:
                HardwareDetector.detect_hardware()
    mock_mps.assert_called_once()
    mock_non_mps.assert_not_called()

def test_detect_hardware_uses_non_mps_otherwise():
    from unittest.mock import patch
    import torch
    with patch.object(torch.backends.mps, "is_available", return_value=False):
        with patch.object(HardwareDetector, "get_device_info_non_mps", return_value={
            "physical_core_count": 4,
            "disk_storage_gb": 500.0,
            "available_disk_storage_gb": 200.0,
            "ram_gb": 8.0,
            "gpus": []
        }) as mock_non_mps:
            with patch.object(HardwareDetector, "get_device_info_mps") as mock_mps:
                HardwareDetector.detect_hardware()
    mock_non_mps.assert_called_once()
    mock_mps.assert_not_called()

