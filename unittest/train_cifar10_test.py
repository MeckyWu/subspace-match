from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
sys.path.insert(0, osp.curdir)

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import models.cifar10
from utils.io_utils import mkdir
from utils.model_utils import convert_state_dict

# script default setting
DATA_ROOT_DIR = './data'
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PIXEL_MEANS = (0.4914, 0.4822, 0.4465)
PIXEL_STDS = (0.247, 0.243, 0.261)


def main():
    # test config
    CKPT_PATH = './output/cifar10/vgg11/rnd_0/best.ckpt'
    MODEL = 'vgg11'

    # create model
    print("=> Creating model '{}'".format(MODEL))
    model = models.cifar10.__dict__[MODEL]()
    assert torch.cuda.is_available(), 'CUDA is required'
    model = model.cuda()

    # load checkpoint
    print("=> Loading checkpoint '{}'".format(CKPT_PATH))
    checkpoint = torch.load(CKPT_PATH)
    print("=> Loaded checkpoint '{}' (epoch {})".format(CKPT_PATH, checkpoint['epoch']))
    assert checkpoint['model'] == MODEL, 'Inconsistent model definition'
    state_dict = convert_state_dict(checkpoint['state_dict'], to_cpu=False)
    model.load_state_dict(state_dict)

    print('=> Preparing data...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEANS, PIXEL_STDS),
    ])
    dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT_DIR, train=False, download=False, transform=transform)

    dataset.test_data = dataset.test_data[0:100, ...]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # switch to evaluate mode
    model.eval()
    output_dir = osp.splitext(__file__)[0] + '_output'
    mkdir(output_dir)

    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        for batch_ind, (input, label) in enumerate(data_loader):
            input = input.cuda(non_blocking=True)

            # forward to get final outputs
            logit = model(input)
            prob = softmax(logit)
            prob = prob.cpu().numpy()[0]
            print(prob, label)

            # get raw image, [H,W,C]
            img = dataset.test_data[batch_ind]
            fname = osp.join(output_dir, '%03d.jpg' % batch_ind)
            print("Plotting prediction %d" % batch_ind)
            plot_pred(img, prob, fname)


def plot_pred(img, prob, fname=None):
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_im, ax_prob = axes
    ax_im.imshow(img)
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    index = [i for i in range(len(CLASSES))]
    ax_prob.bar(index, prob, align='center')
    ax_prob.set_xticks(index)
    ax_prob.set_xticklabels(CLASSES, rotation='vertical', fontsize=10)
    fig.tight_layout()
    if fname is None:
        plt.imshow()
    else:
        fig.savefig(fname)
    plt.close('all')


if __name__ == '__main__':
    main()
