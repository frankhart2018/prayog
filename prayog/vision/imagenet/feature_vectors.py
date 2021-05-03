import torch
import torch.nn as nn
import torchvision.models as models

from .constants import TORCHVISION_VERSION


class PretrainedFeatureVector:
    def __init__(self, model_name):
        self.__model_name = model_name

        self.__model = None

    def __get_pretrained_model(self):
        model = torch.hub.load(f'pytorch/vision:v{TORCHVISION_VERSION}', self.__model_name, pretrained=True)
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)

        return model

    def get_feature_vector(self, input_tensor):
        self.__model = self.__get_pretrained_model() if self.__model == None else self.__model

        output_vector = self.__model(input_tensor)
        output_vector = torch.flatten(output_vector)

        return output_vector