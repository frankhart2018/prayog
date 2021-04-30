import torch.nn as nn

from .layer import Layer


class Conv2d(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        layer_name="conv",
        count=1,
    ):
        super(Conv2d, self).__init__(
            layer=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
            layer_name=layer_name,
            count=count,
        )

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__stride = stride
        self.__padding = padding
        self.__bias = bias
        self.__padding_mode = padding_mode

    def __call__(self, input_tensor):
        return super(Conv2d, self).__call__(input_tensor)

    def __str__(self):
        conv_str = ""

        for count in range(self.count):
            conv_str += (
                "  \033[91m"
                + self.layer_name
                + str(count + 1)
                + "\033[m: "
                + f"prayog.layers.Conv2d(in_channels={self.__in_channels}, "
                + f"out_channels={self.__out_channels}, kernel_size={self.__kernel_size}, "
                + f"stride={self.__stride}, padding={self.__padding}),\n"
            )

        return conv_str
