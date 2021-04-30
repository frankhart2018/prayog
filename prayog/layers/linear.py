import torch.nn as nn

from .layer import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True, layer_name="linear", count=1):
        super(Linear, self).__init__(
            layer=nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            layer_name=layer_name,
            count=count
        )

    def __call__(self, input_tensor):
        return super(Linear, self).__call__(input_tensor)
