from prayog.layers.layer import Layer
import torch.nn as nn
import torch
from torch.nn.modules.linear import Linear

import prayog.layers as layers


model = nn.Sequential(
    layers.Linear(in_features=1, out_features=10, layer_name="fc_1-"),
    layers.Linear(in_features=10, out_features=10, layer_name="fc_2-"),
    layers.Linear(in_features=10, out_features=1, layer_name="fc_3-")
)

print(model)

layer = layers.Linear(
    in_features=10,
    out_features=10,
    layer_name="fc",
    count=1
)

print(layer(torch.randn(10)))