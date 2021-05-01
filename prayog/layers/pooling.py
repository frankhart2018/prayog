import torch.nn as nn

from .layer import Layer


class MaxPool2d(Layer):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        layer_name="maxpool",
        count=1,
    ):
        super(MaxPool2d, self).__init__(
            layer=nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            ),
            layer_name=layer_name,
            count=count,
        )

        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__dilation = dilation
        self.__return_indices = return_indices
        self.__ceil_mode = ceil_mode

    def __call__(self, input_tensor):
        return super().__call__(input_tensor)

    def __str__(self):
        maxpool_str = ""

        for count in range(self.count):
            maxpool_str += (
                "  \033[91m"
                + self.layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.MaxPool2d(kernel_size={self.__kernel_size}, stride={self.__stride}),\n"
            )

        return maxpool_str

    def full_str(self):
        maxpool_str = ""

        for count in range(self.count):
            maxpool_str += (
                "  \033[91m"
                + self.layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.MaxPool2d(kernel_size={self.__kernel_size}, stride={self.__stride}, "
                + f"padding={self.__padding}, dilation={self.__dilation}, return_indices={self.__return_indices}, "
                + f"ceil_mode={self.__ceil_mode}),\n"
            )

        return maxpool_str
