from prayog.layers.layer import Layer
import torch.nn as nn
import torch
from torch.nn.modules.linear import Linear

import prayog.layers as layers
import prayog.models as models

# Sequential models
model = models.Sequential(
    layers.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
    layers.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
)

# print(model)
print(model(torch.randn(1, 1, 28, 28)).shape)

# layer = layers.Linear(
#     in_features=10,
#     out_features=10,
#     layer_name="fc",
#     count=3
# )

# print(layer._Layer__actual_layers)
# print(layer(torch.randn(10)))
# print(layer._Layer__actual_layers)
# print(layer(torch.randn(10)))
