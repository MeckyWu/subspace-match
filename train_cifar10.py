"""
Train CIFAR10 with PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import time
import pprint
import importlib
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import models.cifar10
from utils.train_utils import AverageMeter, adjust_learning_rate, calc_accuracy
from utils.io_utils import mkdir, create_logger

# parse arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config', '-c', required=True,
                    help='model architecture')
parser.add_argument('--rng-seed', type=int, help='random seed')
# optional
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--ckpt-interval', type=int,
                    help='how many epoches to wait before checkpointing training status')
parser.add_argument('--num-worker', type=int, default=1, help='number of workers')
parser.add_argument('--resume', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--eval', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()

# script default setting
DATA_ROOT_DIR = './data'
OUTPUT_ROOT_DIR = './output/cifar10'
PIXEL_MEANS = (0.4914, 0.4822, 0.4465)
PIXEL_STDS = (0.247, 0.243, 0.261)


def main():
    # load config
    cfg = importlib.import_module('configs.cifar10.{}'.format(args.config)).config

    # accuracy
    best_acc = 0
    best_epoch = 0
    start_epoch = 0

    # fix random seed
    if args.rng_seed is not None:
        rng_seed = args.rng_seed
    else:
        rng_seed = cfg.TRAIN.rng_seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    # setup output and logger
    output_dir = mkdir(osp.join(OUTPUT_ROOT_DIR, args.config, 'rnd_%d' % rng_seed))
    logger = create_logger(output_dir)
    logger.info('config:\n' + pprint.pformat(cfg))
    logger.info('arguments:\n' + pprint.pformat(args))
    logger.info('gpu(s): ' + str(os.environ.get('CUDA_VISIBLE_DEVICES')))

    print("=> Creating model '{}'".format(cfg.model))
    model = models.cifar10.__dict__[cfg.model]()
    module_dict = dict(model.named_modules())
    logger.info('module:\n' + pprint.pformat(module_dict))
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # gpu support
    assert torch.cuda.is_available(), 'Training requires cuda'
    # if the input size is fixed, enable it
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    # cudnn.deterministic = True
    # enable DataParallel, default use all cuda devices
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()
    criterion = criterion.cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), cfg.TRAIN.lr,
                                momentum=cfg.TRAIN.momentum, weight_decay=cfg.TRAIN.wd)

    # optionally resume from a checkpoint
    if args.resume:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.eval:
                optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.warn("=> no checkpoint found at '{}'".format(args.resume))

    # load data
    print('=> Preparing data')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # transform into [0.0, 1.0]
        transforms.Normalize(PIXEL_MEANS, PIXEL_STDS),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEANS, PIXEL_STDS),
    ])

    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT_DIR, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.batch_size,
                                               shuffle=True, num_workers=args.num_worker)

    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT_DIR, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TEST.batch_size,
                                             shuffle=False, num_workers=args.num_worker)

    if args.eval:
        logger.info('evaluating trained model')
        acc = validate(val_loader, model, criterion)
        logger.info(
            'Val-Epoch: [{0}]\t'
            'Prec@1: {acc:.3f})'.format(start_epoch, acc=acc)
        )
        return

    def do_checkpoint(epoch, path):
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model': cfg.model,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
        }, path)

    # save initialization state
    do_checkpoint(0, osp.join(output_dir, 'init.ckpt'))

    for epoch in range(start_epoch, cfg.TRAIN.end_epoch):
        adjust_learning_rate(optimizer, epoch, cfg.TRAIN.lr, cfg.TRAIN.lr_step)

        # train for one epoch
        epoch_result = train(train_loader, model, criterion, optimizer, epoch)
        logger.info(
            'Train-Epoch: [{0}]\t'
            'Loss: {loss:.4f}\t'
            'Prec@1: {acc:.3f}'.format(
                epoch + 1, **epoch_result)
        )

        # evaluate on validation set
        acc = validate(val_loader, model, criterion)
        logger.info(
            'Val-Epoch: [{0}]\t'
            'Prec@1: {acc:.3f}'.format(
                epoch + 1, acc=acc)
        )

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        epoch_t = epoch + 1
        if is_best:
            best_epoch = epoch_t
            best_acc = acc
            do_checkpoint(best_epoch, osp.join(output_dir, 'best.ckpt'))
        if (args.ckpt_interval is not None) and (epoch_t % args.ckpt_interval == 0):
            do_checkpoint(epoch_t, osp.join(output_dir, '%03d.ckpt' % epoch_t))

    logger.info(
        '=> Best-Epoch: [{0}]\t'
        'Prec@1: {acc:.3f}'.format(
            best_epoch, acc=best_acc)
    )


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Interface for training
    Args:
        train_loader: data loader
        model: torch.nn.Module
        criterion: loss function
        optimizer:
        epoch: current epoch

    Returns:
        dict: average loss and accuracy

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    toc = time.time()
    for batch_ind, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - toc)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = calc_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - toc)
        toc = time.time()

        if batch_ind % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, batch_ind, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return {'loss': losses.avg, 'acc': top1.avg}


def validate(val_loader, model, criterion):
    """
    Interface for validation
    Args:
        val_loader: data loader
        model: torch.nn.Module
        criterion: loss function

    Returns:
        number: average accuracy

    """
    batch_time = AverageMeter()
    losses = AverageMeter()
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
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = calc_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - toc)
            toc = time.time()

            if batch_ind % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       batch_ind, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()
