"""
Predicate CIFAR10 with PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import argparse
import time
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import models.cifar10
from utils.feature_extractor import FeatureExtractor
from utils.model_utils import convert_state_dict
from utils.train_utils import AverageMeter, calc_accuracy
from utils.io_utils import mkdir, create_logger

# parse arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Predication')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='model architecture')
parser.add_argument('--ckpt', required=True, type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-l', '--layer', dest='target_layer', type=str,
                    help='name of target layer')
parser.add_argument('-t', '--type', dest='target_type', type=str,
                    help='target type')
parser.add_argument('--train', action='store_true',
                    help='whether to use training set')
# optional
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--nb-samples', type=int, help='maximum number of samples to u')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# script default setting
DATA_ROOT_DIR = './data'
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PIXEL_MEANS = (0.4914, 0.4822, 0.4465)
PIXEL_STDS = (0.247, 0.243, 0.261)


def main():
    # setup output and logger
    assert osp.exists(args.ckpt), "Checkpoint path '{}' does not exist.".format(args.ckpt)

    output_path = osp.splitext(args.ckpt)[0]
    output_path = output_path.replace('cifar10', 'cifar10_matrix')
    output_path += '_train' if args.train else '_test'
    output_dir = mkdir(osp.dirname(output_path))

    logger = create_logger(output_dir, log_file=osp.basename(output_path), enable_console=True)
    logger.info('arguments:\n' + pprint.pformat(args))

    print("=> Creating model '{}'".format(args.model))
    model = models.cifar10.__dict__[args.model]()
    assert torch.cuda.is_available(), 'CUDA is required'
    model = model.cuda()

    print("=> Loading checkpoint '{}'".format(args.ckpt))
    checkpoint = torch.load(args.ckpt)
    logger.info("=> Loaded checkpoint '{}' (epoch {})".format(args.ckpt, checkpoint['epoch']))
    assert checkpoint['model'] == args.model, 'Inconsistent model definition'
    # remove module prefix if you use multiple gpus for training
    state_dict = convert_state_dict(checkpoint['state_dict'])
    model.load_state_dict(state_dict)

    print('=> Preparing data...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEANS, PIXEL_STDS),
    ])
    dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT_DIR, train=args.train,
                                           download=False, transform=transform)
    if args.train:
        dataset.train_data = dataset.train_data[:args.nb_samples, ...]  # allow nb_samples to be None
    else:
        dataset.test_data = dataset.test_data[:args.nb_samples, ...]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # sanity check
    acc = validate(data_loader, model)
    logger.info(
        '=> Sanity check\t'
        'Prec@1: {acc:.3f}'.format(acc=acc)
    )

    # setup feature extractor
    feature_extractor = FeatureExtractor(model)

    if args.target_layer is not None:  # assign certain layer
        target_layers = (args.target_layer,)
        target_types = (nn.Module,)
        output_path += '_{}'.format(args.target_layer)
    elif args.target_type is not None:  # assign certain type
        target_layers = feature_extractor.parse_default_layers()
        target_types = feature_extractor.parse_type(args.target_type)
        output_path += '_{}'.format(args.target_type)
    else:
        target_layers = feature_extractor.parse_default_layers()
        target_types = (nn.Module,)

    feature_extractor.append_target_layers(target_layers, target_types)

    logger.info('module:\n' + pprint.pformat(feature_extractor.module_dict))
    logger.info('target_layers:\n' + pprint.pformat(feature_extractor.target_outputs.keys()))

    predicate(data_loader, feature_extractor, output_path)
    print("=> finish")


def predicate(data_loader, feature_extractor, output_path=None):
    batch_time = AverageMeter()
    model = feature_extractor.model
    outputs_dict = dict()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        toc = time.time()
        for batch_ind, (input, _) in enumerate(data_loader):
            input = input.cuda(non_blocking=True)

            # forward to get intermediate outputs
            _ = model(input)

            # synchronize so that everything is calculated
            torch.cuda.synchronize()

            # print(feature_extractor.target_outputs)
            for target_layer, target_output in feature_extractor.target_outputs.items():
                if target_layer in outputs_dict:
                    outputs_dict[target_layer].append(target_output.data.numpy())
                else:
                    outputs_dict[target_layer] = [target_output.data.numpy()]

            # measure elapsed time
            batch_time.update(time.time() - toc)
            toc = time.time()

            if batch_ind % args.log_interval == 0:
                print('Predicate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_ind, len(data_loader), batch_time=batch_time))

    if output_path is not None:
        def _squeeze_dict(d):
            for key, val in d.items():
                d[key] = np.concatenate(val, 0)
            return d

        outputs_dict = _squeeze_dict(outputs_dict)
        np.savez_compressed(output_path, **outputs_dict)


def validate(val_loader, model):
    """
    Interface for validating model
    Args:
        val_loader: data loader
        model: nn.Module

    Returns:
        number: top1 accuracy
    """
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        toc = time.time()
        for batch_ind, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1 = calc_accuracy(output, target, topk=(1,))
            top1.update(prec1[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - toc)
            toc = time.time()

            if batch_ind % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       batch_ind, len(val_loader), batch_time=batch_time, top1=top1))

        # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()
