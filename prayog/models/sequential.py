class Sequential:
    def __init__(self, *args):
        self.__layers = args

    def __call__(self, input_tensor):
        out = input_tensor
        prev_layer = None

        for layer_number, layer in enumerate(self.__layers):
            try:
                out = layer(out)
            except RuntimeError as re:
                layer.incompatible_shape_input(shape=out.size(), layer_number=layer_number)
                return None

        return out

    def __str__(self):
        sequential_str = "prayog.models.Sequential(\n"

        for layer in self.__layers:
            sequential_str += str(layer)

        sequential_str += ")"

        return sequential_str
