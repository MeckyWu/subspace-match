import torch
import torch.nn as nn

from models.cifar10 import NUM_CHANNELS, NUM_CLASSES, IMAGE_SIZE
from utils.model_utils import init_conv_weights, init_fc_weights, init_bn_weights


__all__ = ['plain20', 'plain32', 'plain34', 'plain42', 'plain44', 'plain56']


class PlainNet(nn.Module):
    def __init__(self, num_block, num_filters=(16, 32, 64)):
        super(PlainNet, self).__init__()
        self._initialized = False
        self.features = self._make_layers(num_block, num_filters)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], NUM_CLASSES),
        )
        pool_size = int(IMAGE_SIZE[0] / (2**(len(num_filters)-1)))
        self.global_pool = nn.AvgPool2d(pool_size, stride=1)
        self._init_weights()

    def forward(self, x):
        assert self._initialized, "{} without explicitly initialized".format(str(self.__class__))
        out = self.features(x)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, n, dims):
        layers = []
        in_channels = NUM_CHANNELS
        x = dims[0]
        for ind in range(2*n+1):
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x

        for x in dims[1:]:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=2),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
            for _ in range(2*n-1):
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

    def _init_weights(self):
        init_conv_weights(self)
        init_fc_weights(self, std=0.01)
        init_bn_weights(self)
        self._initialized = True


class plain20(PlainNet):
    def __init__(self):
        super(plain20, self).__init__(3)


class plain32(PlainNet):
    def __init__(self):
        super(plain32, self).__init__(5)


class plain34(PlainNet):
    def __init__(self):
        super(plain34, self).__init__(4, (16, 32, 64, 128))


class plain42(PlainNet):
    def __init__(self):
        super(plain42, self).__init__(5, (16, 32, 64, 128))


class plain44(PlainNet):
    def __init__(self):
        super(plain44, self).__init__(7)
        self._init_weights()


class plain56(PlainNet):
    def __init__(self):
        super(plain56, self).__init__(9)
        self._init_weights()


def unittest():
    net = plain20()
    x = torch.randn(2, NUM_CHANNELS, 32, 32)
    y = net(x)
    print("output size:", y.size())
    print("output value:", y.data.numpy())


if __name__ == '__main__':
    unittest()