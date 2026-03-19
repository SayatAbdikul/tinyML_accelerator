""" loads, evaluates and runs the pretrained model"""
import torch
import torch.nn as nn
from accelerator_config import AcceleratorConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
model = None
def create_mlp_model():
        # 1. Define the MLP model
    class Digit_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, 12)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(12, 32)
            self.fc3 = nn.Linear(32, AcceleratorConfig.OUT_N)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    # 2. Set model parameters
    input_size = 28 * 28
    # 3. Initialize model and set to eval mode
    global model
    model = Digit_Model()
    model.load_state_dict(torch.load("digit_model_weights.pth", map_location=torch.device('cpu'))) # Load pre-trained weights        
    # 4. Create a dummy input tensor for ONNX export
    dummy_input = torch.randn(1, 1, 28, 28)

    # 5. Export the model to ONNX format
    onnx_filename = "mlp_model.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_filename,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18
    )

    # print(f"MLP model successfully exported to '{onnx_filename}'")


def create_cnn_model():
    """Define, export and save a small LeNet-style CNN for MNIST.

    Architecture (all ops supported by the CNN ISA):
        Input : 1 × 28 × 28
        Conv1 : 4 filters, 3×3, no-pad, stride 1  → [4, 26, 26]
        ReLU
        MaxPool: 2×2, stride 2                     → [4, 13, 13] = 676 elems
        Conv2 : 8 filters, 3×3, no-pad, stride 1  → [8, 11, 11]
        ReLU
        MaxPool: 2×2, stride 2                     → [8,  5,  5] = 200 elems
        Flatten                                    → 200
        FC    : 200 → 10

    Weight budget: Conv1=36, Conv2=288, FC=2000  ≈ 2.3 KB
    """
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1   = nn.Conv2d(1, 4,  kernel_size=3, stride=1, padding=0, bias=True)
            self.relu    = nn.ReLU()
            self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2   = nn.Conv2d(4, 8,  kernel_size=3, stride=1, padding=0, bias=True)
            self.fc      = nn.Linear(8 * 5 * 5, AcceleratorConfig.OUT_N, bias=True)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))   # → [4, 13, 13]
            x = self.pool(self.relu(self.conv2(x)))   # → [8,  5,  5]
            x = x.view(x.size(0), -1)                 # flatten → 200
            return self.fc(x)

    global model
    model = SmallCNN()
    # Use random (untrained) weights so the ONNX export works out-of-the-box.
    # Replace with model.load_state_dict(torch.load("cnn_weights.pth")) to use trained weights.
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_filename = "cnn_model.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_filename,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,   # opset 11 keeps MaxPool simple (no ceil_mode by default)
    )
    print(f"SmallCNN exported to '{onnx_filename}'")
    return model

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get index of max log-probability

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def run_model():
    global model
    if model is None:
        raise ValueError("Model has not been created. Call create_mlp_model() first.")
        
    # 1. Define transformations (e.g., convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to Tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std from MNIST dataset
    ])

    # 2. Load the training and test datasets
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. Wrap them in DataLoader for batching and shuffling
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return evaluate_model(model, test_loader=test_loader, device='cpu')

if __name__ == "__main__":
    # 1. Create and save the model
    create_mlp_model()

    # 2. Run the model
    run_model()