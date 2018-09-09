import torch
import torch.nn as nn


def convert_state_dict(ori_state_dict, to_cpu=True):
    """
    Convert data parallel gpu trained model to single gpu/cpu model
    Args:
        ori_state_dict: original state dict
        to_cpu: bool, whether to convert to cpu model

    Returns:
        dict: new converted state dict

    """
    new_state_dict = dict()
    for key, val in ori_state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = val.cpu() if to_cpu else val
        else:
            new_state_dict[key] = val.cpu() if to_cpu else val
    return new_state_dict


def init_conv_weights(model, mode="kaiming", nonlinearity='relu', **kwargs):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, nonlinearity=nonlinearity)
            elif mode == "xaiver":
                nn.init.xavier_normal_(m.weight.data)
            elif mode == "normal":
                nn.init.normal_(m.weight.data, mean=kwargs['mean'], std=kwargs['std'])
            elif mode == "uniform":
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity=nonlinearity)
            else:
                raise ValueError("Unknown mode")

            if m.bias is not None:
                if 'bias' in kwargs:
                    m.bias.data.fill_(kwargs['bias'])
                else:
                    m.bias.data.zero_()


def init_fc_weights(model, mode="normal", **kwargs):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if mode == "normal":
                m.weight.data.normal_(0, kwargs['std'])
            elif mode == "uniform":
                m.weight.data.uniform_(-kwargs['std'], kwargs['std'])
            elif mode == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, nonlinearity=kwargs['nonlinearity'])
            elif mode == "xaiver":
                nn.init.xavier_normal_(m.weight.data)

            if m.bias is not None:
                if 'bias' in kwargs:
                    m.bias.data.fill_(kwargs['bias'])
                else:
                    m.bias.data.zero_()


def init_bn_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
