"""
Unit tests for device utilities.
Tests device resolution, availability checking, and device selection logic.
"""
import unittest
from unittest.mock import Mock, patch


class TestDeviceUtils(unittest.TestCase):
    """Test device utility functions."""

    @patch('ModelForge.utilities.device_utils.torch')
    def test_resolve_device_auto_prefers_cuda(self, mock_torch):
        """Test that auto device resolution prefers CUDA when available."""
        from ModelForge.utilities.device_utils import resolve_device
        
        # Setup: CUDA is available
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla V100"
        mock_torch.device = Mock(side_effect=lambda x: f"device:{x}")
        
        device_obj, device_type = resolve_device("auto")
        
        self.assertEqual(device_type, "cuda")
        mock_torch.cuda.is_available.assert_called()

    @patch('ModelForge.utilities.device_utils.torch')
    def test_resolve_device_auto_falls_back_to_mps(self, mock_torch):
        """Test that auto device resolution falls back to MPS when CUDA unavailable."""
        from ModelForge.utilities.device_utils import resolve_device
        
        # Setup: CUDA not available, MPS is available
        mock_torch.cuda.is_available.return_value = False
        mock_backends = Mock()
        mock_backends.mps.is_available.return_value = True
        mock_torch.backends = mock_backends
        mock_torch.device = Mock(side_effect=lambda x: f"device:{x}")
        
        device_obj, device_type = resolve_device("auto")
        
        self.assertEqual(device_type, "mps")

    @patch('ModelForge.utilities.device_utils.torch')
    def test_resolve_device_cuda_requires_availability(self, mock_torch):
        """Test that requesting CUDA when unavailable raises error."""
        from ModelForge.utilities.device_utils import resolve_device
        from ModelForge.exceptions import ConfigurationError
        
        # Setup: CUDA not available
        mock_torch.cuda.is_available.return_value = False
        
        with self.assertRaises(ConfigurationError) as context:
            resolve_device("cuda")
        
        self.assertIn("CUDA", str(context.exception))

    @patch('ModelForge.utilities.device_utils.torch')
    def test_is_device_available_mps(self, mock_torch):
        """Test MPS availability checking."""
        from ModelForge.utilities.device_utils import is_device_available
        
        mock_backends = Mock()
        mock_backends.mps.is_available.return_value = True
        mock_torch.backends = mock_backends
        
        self.assertTrue(is_device_available("mps"))


if __name__ == '__main__':
    unittest.main()
