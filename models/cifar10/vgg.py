import torch
import torch.nn as nn

from models.cifar10 import NUM_CHANNELS, NUM_CLASSES
from utils.model_utils import init_conv_weights, init_fc_weights, init_bn_weights

__all__ = ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
           'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
           ]

cfg = {
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, dim_fc=512, batch_norm=False, dropout=False):
        super(VGG, self).__init__()
        self._initialized = False
        vgg_config = cfg[vgg_name]
        dim_conv = vgg_config[-2]
        self.features = self._make_conv_layers(vgg_config, batch_norm)
        self.classifier = self._make_fc_layers(dim_conv, dim_fc, dropout)
        self._init_weights()  # overwrite if necessary

    def forward(self, x):
        assert self._initialized, "{} without explicitly initialized".format(str(self.__class__))
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_conv_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = NUM_CHANNELS
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_fc_layers(self, dim_conv, dim_fc, dropout=False):
        if dropout:
            return nn.Sequential(
                nn.Linear(dim_conv, dim_fc),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(dim_fc, dim_fc),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(dim_fc, NUM_CLASSES),
            )
        else:
            return nn.Sequential(
                nn.Linear(dim_conv, dim_fc),
                nn.ReLU(True),
                nn.Linear(dim_fc, dim_fc),
                nn.ReLU(True),
                nn.Linear(dim_fc, NUM_CLASSES),
            )

    def _init_weights(self):
        init_conv_weights(self)
        init_fc_weights(self, std=0.01)
        init_bn_weights(self)
        self._initialized = True


class vgg8(VGG):
    def __init__(self):
        super(vgg8, self).__init__('vgg8')


class vgg11(VGG):
    def __init__(self):
        super(vgg11, self).__init__('vgg11')


class vgg13(VGG):
    def __init__(self):
        super(vgg13, self).__init__('vgg13')


class vgg16(VGG):
    def __init__(self):
        super(vgg16, self).__init__('vgg16')


class vgg19(VGG):
    def __init__(self):
        super(vgg19, self).__init__('vgg19')


class vgg11_bn(VGG):
    def __init__(self):
        super(vgg11_bn, self).__init__('vgg11', batch_norm=True)


class vgg13_bn(VGG):
    def __init__(self):
        super(vgg13_bn, self).__init__('vgg13', batch_norm=True)


class vgg16_bn(VGG):
    def __init__(self):
        super(vgg16_bn, self).__init__('vgg16', batch_norm=True)


def unittest():
    from models.cifar10 import IMAGE_SIZE
    net = vgg11()
    x = torch.randn(2, NUM_CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1])
    y = net(x)
    print("output size:", y.size())
    print("output value:", y.data.numpy())


if __name__ == '__main__':
    unittest()
