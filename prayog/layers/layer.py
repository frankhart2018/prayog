import torch.nn as nn
from collections import OrderedDict


class Layer:
    def __init__(self, layer, layer_name, count):
        self.__layer = layer
        self.__layer_name = layer_name
        self.__count = count

        self.__actual_layers = None

    @property
    def layer(self):
        return self.__layer

    @property
    def layer_name(self):
        return self.__layer_name

    @property
    def count(self):
        return self.__count

    def __get_layer_ordered_dict(self):
        return OrderedDict(
            {self.__layer_name + str(i + 1): self.__layer for i in range(self.__count)}
        )

    def __call__(self, input_tensor):
        self.__actual_layers = (
            self.__get_layer_ordered_dict()
            if self.__actual_layers == None
            else self.__actual_layers
        )

        out = input_tensor

        for layer_name, layer in self.__actual_layers.items():
            out = layer(out)

        return out
