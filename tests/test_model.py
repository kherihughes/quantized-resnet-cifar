import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import create_resnet18

class TestResNetModel(unittest.TestCase):
    def setUp(self):
        self.model = create_resnet18()
        self.device = torch.device('cpu')
        
    def test_model_creation(self):
        """Test if model can be created."""
        self.assertIsNotNone(self.model)
        
    def test_model_forward(self):
        """Test if model can perform forward pass."""
        x = torch.randn(1, 3, 32, 32)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))
        
    def test_model_parameters(self):
        """Test if model has correct number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)

if __name__ == '__main__':
    unittest.main()