import torch
import torch.nn as nn

# 1. Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# 2. Set model parameters
input_size = 28 * 28       # e.g. for MNIST images
hidden_size = 128
output_size = 10           # number of classes

# 3. Initialize model and set to eval mode
model = MLP(input_size, hidden_size, output_size)
model.eval()

# 4. Create dummy input for ONNX export
dummy_input = torch.randn(1, 1, 28, 28)  # shape: [batch_size, channels, height, width]

# 5. Export the model to ONNX format
onnx_filename = "mlp_model.onnx"
torch.onnx.export(
    model, dummy_input, onnx_filename,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"MLP model successfully exported to '{onnx_filename}'")
