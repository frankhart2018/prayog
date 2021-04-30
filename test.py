from prayog.layers.layer import Layer
import torch.nn as nn
import torch
from torch.nn.modules.linear import Linear

import prayog.layers as layers
import prayog.models as models

# Sequential models
model = models.Sequential(
    layers.Linear(in_features=1, out_features=10, layer_name="fc1-"),
    layers.Linear(in_features=10, out_features=10, layer_name="fc2-", count=3),
    layers.Linear(in_features=10, out_features=1, layer_name="fc3-"),
)

print(model)
print(model(torch.randn(1)))

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
