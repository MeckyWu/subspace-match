from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
sys.path.insert(0, osp.curdir)

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from utils.model_utils import init_fc_weights
from utils.feature_extractor import FeatureExtractor


class Net(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters):
        super(Net, self).__init__()
        self._initialized = False
        self.num_layers = num_layers
        self.num_filters = num_filters

        layers = []
        for layer_idx in range(num_layers):
            layers += [nn.Linear(in_channels, num_filters, bias=True), nn.ReLU(inplace=True)]
            in_channels = num_filters

        self.layers = layers
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        assert self._initialized, "{} without explicitly initialized".format(str(self.__class__))
        # outputs = []
        for layer in self.layers:
            x = layer(x)
            # outputs.append(x.cpu().numpy())
        # return outputs
        return x

    def partial_forward(self, x, index):
        assert self._initialized, "{} without explicitly initialized".format(str(self.__class__))
        for layer_ind, layer in enumerate(self.layers):
            if layer_ind <= index:
                continue
            x = layer(x)
        return x


def main():
    # build model
    in_channels = 3
    num_samples = 16
    num_layers = 16
    num_filter = 256

    net = Net(in_channels, num_layers, num_filter)
    net.cuda()
    init_fc_weights(net, std=0.07)
    net._initialized = True

    data = torch.Tensor(num_samples, in_channels)
    data.normal_(mean=0, std=1)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    feature_extractor = FeatureExtractor(net, True)
    layer_names = feature_extractor.parse_default_layers()
    feature_extractor.append_target_layers(layer_names)

    with torch.no_grad():
        for data_batch in data_loader:
            intput = data_batch[0]
            intput = intput.cuda(non_blocking=True)

            output = net(intput)
            outputs = feature_extractor.target_outputs
            for index, layer_name in enumerate(layer_names):
                feat = outputs[layer_name]
                feat = feat.cuda(non_blocking=True)
                np_output = net.partial_forward(feat, index)
                np_output = np_output.cpu().numpy()
                np.testing.assert_array_almost_equal(output, np_output)
                print("Pass %d" % index)


if __name__ == '__main__':
    main()
