import unittest
import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import create_resnet18

class TestInference(unittest.TestCase):
    def setUp(self):
        self.model = create_resnet18()
        self.model.eval()
        
    def test_inference_speed(self):
        """Test if inference time is within acceptable range."""
        x = torch.randn(1, 3, 32, 32)
        
        # Warmup
        for _ in range(5):
            self.model(x)
        
        # Measure time
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                self.model(x)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) * 10  # ms per inference
        self.assertLess(avg_time, 50)  # Should be less than 50ms
        
    def test_batch_inference(self):
        """Test if model can handle different batch sizes."""
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 32, 32)
            with torch.no_grad():
                output = self.model(x)
            self.assertEqual(output.shape, (batch_size, 10))

if __name__ == '__main__':
    unittest.main()