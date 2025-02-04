import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import create_resnet18

def load_and_preprocess_image(image_path):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    image = Image.open(image_path)
    return transform(image)

def run_inference(model, image):
    """Run inference and measure time."""
    model.eval()
    with torch.no_grad():
        start_time = time.perf_counter()
        output = model(image.unsqueeze(0))
        inference_time = (time.perf_counter() - start_time) * 1000
        probs = torch.nn.functional.softmax(output, dim=1)
    return probs.squeeze(), inference_time

def main():
    # Load models
    device = torch.device('cpu')
    orig_model = create_resnet18()
    orig_model.load_state_dict(torch.load('models/resnet18_cifar10.pth', map_location=device))
    
    quant_model = create_resnet18()
    quant_model.eval()
    quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(quant_model, inplace=True)
    quant_model.load_state_dict(torch.load('models/quantized_resnet18_cifar10.pth', map_location=device))
    
    # Run demo
    image_path = input("Enter image path: ")
    image = load_and_preprocess_image(image_path)
    
    orig_probs, orig_time = run_inference(orig_model, image)
    quant_probs, quant_time = run_inference(quant_model, image)
    
    # Print results
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nResults:")
    print(f"Original Model (took {orig_time:.2f}ms):")
    for i, prob in enumerate(orig_probs):
        print(f"{classes[i]}: {prob*100:.2f}%")
    
    print(f"\nQuantized Model (took {quant_time:.2f}ms):")
    for i, prob in enumerate(quant_probs):
        print(f"{classes[i]}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()