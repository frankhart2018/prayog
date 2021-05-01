from prayog.layers import layer
from prayog.layers.layer import Layer
import torch.nn as nn
import torch
from torch.nn.modules.linear import Linear

import prayog.layers as layers
import prayog.models as models

# Sequential models
model = models.Sequential(
    layers.Conv2d(in_channels=3, out_channels=5, kernel_size=3),
    layers.MaxPool2d(kernel_size=2, stride=2),
    layers.Flatten(),
    layers.Linear(in_features="auto", out_features=10),
)

print(model.longer_print())
# print(model(torch.randn(1)))