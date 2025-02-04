import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import create_resnet18

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = create_resnet18()
        self.model.eval()
        
    def test_quantization_preparation(self):
        """Test if model can be prepared for quantization."""
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        self.assertTrue(hasattr(self.model, 'qconfig'))
        
    def test_model_conversion(self):
        """Test if model can be converted to quantized version."""
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        self.assertTrue(hasattr(self.model, '_is_stateless'))

if __name__ == '__main__':
    unittest.main()