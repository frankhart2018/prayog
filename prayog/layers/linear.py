import torch.nn as nn
from torch.nn.modules import linear

from .layer import Layer


class Linear(Layer):
    def __init__(
        self, in_features, out_features, bias=True, layer_name="linear", count=1
    ):
        super(Linear, self).__init__(
            layer=nn.Linear(
                in_features=in_features, out_features=out_features, bias=bias
            ),
            layer_name=layer_name,
            count=count,
        )

        self.__in_features = in_features
        self.__out_features = out_features
        self.__bias = bias
        self.__layer_name = layer_name
        self.__count = count

    def __call__(self, input_tensor):
        return super(Linear, self).__call__(input_tensor)

    def __str__(self):
        linear_str = ""

        for count in range(self.__count):
            linear_str += (
                "  \033[91m"
                + self.__layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.Linear(in_features={self.__in_features}, out_features={self.__out_features}, bias={self.__bias}),\n"
            )

        return linear_str
