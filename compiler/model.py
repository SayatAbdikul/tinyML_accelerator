import torch
import torch.nn as nn
model = None
def create_mlp_model():
        # 1. Define the MLP model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten input
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # 2. Set model parameters
    input_size = 10 * 10       # smaller to have a manageable model
    hidden_size = 128
    output_size = 10           # number of classes

    # 3. Initialize model and set to eval mode
    global model
    model = MLP(input_size, hidden_size, output_size)
    model.eval()

    # 4. Create dummy input for ONNX export
    dummy_input = torch.randn(1, 1, 10, 10)  # shape: [batch_size, channels, height, width]

    # 5. Export the model to ONNX format
    onnx_filename = "mlp_model.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_filename,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

    # print(f"MLP model successfully exported to '{onnx_filename}'")


def run_model(input_tensor):
    """Run the MLP model with the given input tensor."""
    if model is None:
        raise ValueError("Model has not been created. Call create_mlp_model() first.")
    
    # Ensure input tensor is in the correct shape
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if missing

    with torch.no_grad():
        output = model(input_tensor)
    
    return output