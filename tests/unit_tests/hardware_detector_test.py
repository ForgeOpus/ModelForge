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

    
