import prayog.layers as layers


layer = layers.Linear(
    in_features=10,
    out_features=10,
    layer_name="fc",
    count=1
)

import torch

print(layer(torch.randn(10)))