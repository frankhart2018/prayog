import torch.nn as nn

from prayog.utils import error
import prayog.layers as layers


class Sequential:
    def __init__(self, *args):
        self.__layers = list(args)
        self.__params = []

        for i in range(len(self.__layers)):
            if i > 0 and self.__layers[i].__class__.__name__ == "Linear" and self.__layers[i].in_features == "auto":
                in_features = 0
                out_features = self.__layers[i].out_features
                bias = self.__layers[i].bias

                layer_name = self.__layers[i].layer_name
                count = self.__layers[i].count

                if self.__layers[i-1].__class__.__name__ == "Linear":
                    in_features = self.__layers[i-1].out_features
                elif self.__layers[i-1].__class__.__name__ in ["Conv2d", "MaxPool2d", "Flatten"]:
                    error.throw(
                        error_type="IncorrectAutoSpecificationError",
                        error_msg=f"Cannot specify auto without providing Input shape when using {self.__layers[i-1].__class__.__name__}"
                    )

                self.__layers[i] = layers.Linear(in_features=in_features, out_features=out_features, bias=bias,
                                                 layer_name=layer_name, count=count)

            self.__params.extend(self.__layers[i].layer.parameters())

    def __call__(self, input_tensor):
        out = input_tensor
        prev_layer = None

        for layer_number, layer in enumerate(self.__layers):
            try:
                out = layer(out)
            except RuntimeError as re:
                layer.incompatible_shape_input(
                    shape=out.size(),
                    layer_number=layer_number + 1,
                    prev_layer_type=prev_layer.__class__.__name__,
                )
                return None

            prev_layer = layer

        return out

    def __str__(self):
        sequential_str = "prayog.models.Sequential(\n"

        for layer in self.__layers:
            sequential_str += str(layer)

        sequential_str += ")"

        return sequential_str

    def longer_print(self):
        sequential_str = "prayog.models.Sequential(\n"

        for layer in self.__layers:
            sequential_str += layer.full_str()

        sequential_str += ")"

        return sequential_str

    def parameters(self):
        for param in self.__params:
            yield param
