import torch.nn as nn

from .layer import Layer


class Flatten(Layer):
    def __init__(self, start_dim=1, end_dim=-1, layer_name="flatten", count=1):
        super(Flatten, self).__init__(
            layer=nn.Flatten(start_dim=start_dim, end_dim=end_dim),
            layer_name=layer_name,
            count=count,
        )

        self.__start_dim = start_dim
        self.__end_dim = end_dim

    def __call__(self, input_tensor):
        return super().__call__(input_tensor)

    def __str__(self):
        flatten_str = ""

        for count in range(self.count):
            flatten_str += (
                "  \033[91m"
                + self.layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.Flatten(),\n"
            )

        return flatten_str

    def full_str(self):
        flatten_str = ""

        for count in range(self.count):
            flatten_str += (
                "  \033[91m"
                + self.layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.Flatten(start_dim={self.__start_dim}, end_dim={self.__end_dim}),\n"
            )

        return flatten_str
