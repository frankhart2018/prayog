import torch.nn as nn
from torch.nn.modules import linear

from .layer import Layer
from prayog.utils import error


class Linear(Layer):
    def __init__(
        self, in_features, out_features, bias=True, layer_name="linear", count=1
    ):
        if count > 1 and in_features != "auto" and in_features != out_features:
            error.throw(
                error_type="IncorrectLinearLayerError",
                error_msg="in_features should be equal to out_features when count is greater than 1",
            )

        mock_in_features = in_features if in_features != "auto" else 1
        super(Linear, self).__init__(
            layer=nn.Linear(
                in_features=mock_in_features, out_features=out_features, bias=bias
            ),
            layer_name=layer_name,
            count=count,
        )

        self.__in_features = in_features
        self.__out_features = out_features
        self.__bias = bias

    @property
    def in_features(self):
        return self.__in_features

    @property
    def out_features(self):
        return self.__out_features

    @property
    def bias(self):
        return self.__bias

    def __call__(self, input_tensor):
        return super(Linear, self).__call__(input_tensor)

    def __str__(self):
        linear_str = ""

        for count in range(self.count):
            linear_str += (
                "  \033[91m"
                + self.layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.Linear(in_features={self.__in_features}, out_features={self.__out_features}, bias={self.__bias}),\n"
            )

        return linear_str

    def full_str(self):
        return self.__str__()

    def incompatible_shape_input(self, shape, layer_number, prev_layer_type):
        expected_shape = shape[-1]

        if prev_layer_type in ["Conv2d", "MaxPool2d"]:
            error.throw(
                error_type="IncorrectLinearLayerError",
                error_msg=f"Cannot pass feature maps from {prev_layer_type} without flattening to Linear layer",
            )

        error.throw(
            error_type="IncorrectLinearLayerError",
            error_msg=f"Expected in_features to be {expected_shape}, but got {self.__in_features} at layer/block #{layer_number} {self.layer_name}",
        )
