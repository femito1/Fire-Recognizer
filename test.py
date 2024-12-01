import torch
import sys

print(sys.executable)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)