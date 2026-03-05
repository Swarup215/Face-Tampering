import torch
from mesonet import MesoNet

model = MesoNet()
dummy_input = torch.randn(1, 3, 256, 256)

output = model(dummy_input)
print("Output shape:", output.shape)
print("Output value:", output.item())