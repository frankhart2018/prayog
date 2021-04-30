class Sequential:
    def __init__(self, *args):
        self.__layers = args

    def __call__(self, input_tensor):
        out = input_tensor

        for layer in self.__layers:
            out = layer(out)

        return out

    def __str__(self):
        sequential_str = "prayog.models.Sequential(\n"

        for layer in self.__layers:
            sequential_str += str(layer)

        sequential_str += ")"

        return sequential_str
