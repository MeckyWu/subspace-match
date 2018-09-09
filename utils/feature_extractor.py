from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import torch
import torch.nn as nn


class FeatureExtractor(object):
    def __init__(self, model, is_cuda=True):
        self.model = model
        self.target_layers = list()
        self.target_outputs = dict()
        self.is_cuda = is_cuda

    @property
    def module_dict(self):
        return dict(self.model.named_modules())

    def parse_default_layers(self):
        """
        Assume that the model consists of features and classifier,
        which are composed of nn.Sequential
        Returns:
            list: names of layers

        """
        layer_names = []

        try:
            num_features = len(list(self.model.features))
        except:
            Warning('No features')
            num_features = 0

        try:
            num_classifier = len(list(self.model.classifier))
        except:
            Warning('No classifiers')
            num_classifier = 0

        for i in range(num_features):
            layer_name = 'features.{}'.format(i)
            layer_names.append(layer_name)

        for i in range(num_classifier):
            layer_name = 'classifier.{}'.format(i)
            layer_names.append(layer_name)

        return layer_names

    @staticmethod
    def parse_type(t=None):
        if t is None:
            return (nn.Module,)
        elif t == 'conv':
            return (nn.Conv2d,)
        elif t == 'relu':
            return (nn.ReLU,)
        elif t == 'fc':
            return (nn.Linear,)
        elif t == 'pool':
            return (nn.MaxPool2d, nn.AvgPool2d)
        elif t == 'bn':
            return (nn.BatchNorm2d,)
        elif t == 'block':
            return (nn.Sequential,)
        else:
            raise ValueError("Unknown type")

    def append_target_layers(self, layer_names, type_filters=None):
        """
        Bind hook functions for certain layers to get intermediate outputs
        Args:
            layer_names (tuple/list): names of target layers
            type_filters (tuple/list): target types

        Returns:

        """
        assert isinstance(layer_names, (tuple, list))

        def hook_func(module, input, output, layer_name):
            """
            Because some replace operator, we have to get the result before being replaced
            """
            # print(layer_name + 'hook')
            if self.is_cuda:
                self.target_outputs[layer_name] = output.cpu()
            else:
                self.target_outputs[layer_name] = output.clone()

        module_dict = self.module_dict
        for layer_name in layer_names:
            layer = module_dict[layer_name]
            if type_filters is not None and not isinstance(layer, type_filters):
                continue

            forward_hook = partial(hook_func, layer_name=layer_name)

            self.target_layers.append(layer)
            self.target_outputs[layer_name] = list()
            layer.register_forward_hook(forward_hook)
