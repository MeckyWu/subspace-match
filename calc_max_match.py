"""
Calculate maximal match for a given epsilon
"""

import os.path as osp
import argparse
import pprint
import time
from collections import OrderedDict
import numpy as np


from utils.match_utils import find_maximal_match, find_maximal_epsilon
from utils.io_utils import mkdir, write_pkl, create_logger


parser = argparse.ArgumentParser(description='Calculate maximum match.')
parser.add_argument('-e', '--eps', type=float,
                    help="epsilon for maximal match")
parser.add_argument('--mat-dir', type=str, required=True,
                    help='directory contains output matrices')
parser.add_argument('--mat-prefix', type=str, default='best_test_relu',
                    help='prefix of matrix')
parser.add_argument('--mat-prefix2', type=str,
                    help='prefix of matrix2')
parser.add_argument('--s0', type=int, default=0,
                    help="random seed 0")
parser.add_argument('--s1', type=int, default=1,
                    help="random seed 1")
# optional
parser.add_argument('-l', '--layer', dest='target_layer', type=str,
                    help='name of target layer')
parser.add_argument('--nb-samples', type=int,
                    help="number of samples")
parser.add_argument('--ndim', dest='sample_ndim', type=int, default=10000,
                    help="only for feature maps")
parser.add_argument('--iter', dest='sample_iter', type=int, default=16,
                    help="number of samples")
args = parser.parse_args()


def main():
    # fix random seed
    np.random.seed(0)

    # aliases
    nb_samples = args.nb_samples
    sample_ndim = args.sample_ndim
    sample_iter = args.sample_iter
    mat_root_dir = args.mat_dir
    seed0 = args.s0
    seed1 = args.s1
    mat_prefix = args.mat_prefix
    mat_prefix2 = mat_prefix if args.mat_prefix2 is None else args.mat_prefix2
    epsilon = args.eps

    # setup output and logger
    output_dir = mat_root_dir.replace('matrix', 'max_match')
    output_dir = osp.join(output_dir, mat_prefix, 'rnd_{}_{}'.format(seed0, seed1))
    output_dir = mkdir(output_dir)
    logger = create_logger(output_dir, log_file='eps_{:.3f}'.format(epsilon), enable_console=True)
    logger.info('arguments:\n' + pprint.pformat(args))

    # load features
    mat_path0 = osp.join(mat_root_dir, 'rnd_{}'.format(seed0), mat_prefix + '.npz')
    mat_path1 = osp.join(mat_root_dir, 'rnd_{}'.format(seed1), mat_prefix2 + '.npz')

    logger.info('Loading mat_path0: {}'.format(mat_path0))
    mat_dict0 = np.load(mat_path0)
    logger.info('Loading mat_path1: {}'.format(mat_path1))
    mat_dict1 = np.load(mat_path1)

    if args.target_layer is None:
        layers = mat_dict0.keys()
    else:
        layers = [args.target_layer]
    logger.info('target layers: {}'.format(layers))

    match_dict = OrderedDict()
    for layer in layers:
        logger.info('=> Calculating layer {} ...'.format(layer))

        # lazy load
        tic = time.time()
        mat0 = mat_dict0[layer]
        mat1 = mat_dict1[layer]
        toc = time.time()
        print('Loaded data with {:.2f}s'.format(toc - tic))

        # reshape
        mat0 = mat0[:nb_samples, ...]
        mat1 = mat1[:nb_samples, ...]
        # mat1 = mat1[np.random.permutation(nb_samples), ...]
        logger.info("mat0 with shape {}".format(mat0.shape))
        logger.info("mat1 with shape {}".format(mat1.shape))

        if 'classifier' in layer:
            mat0 = mat0.reshape([mat0.shape[0], -1]).transpose()
            mat1 = mat1.reshape([mat1.shape[0], -1]).transpose()
            assert mat0.shape[1] == mat1.shape[1], 'Check the dimension of each feature vector'

            # # check norm of features
            # norm0 = np.linalg.norm(mat0, axis=1)
            # norm1 = np.linalg.norm(mat1, axis=1)
            # mean_norm0 = np.mean(norm0) / np.sqrt(mat0.shape[1])
            # mean_norm1 = np.mean(norm1) / np.sqrt(mat1.shape[1])
            # logger.info("mean norm0 = {:.3f}, mean norm1 = {:.3f}".format(mean_norm0, mean_norm1))

            # # check how many neurons are inactive
            # norm0 = np.linalg.norm(mat0, axis=0) / np.sqrt(len(mat0))
            # norm1 = np.linalg.norm(mat1, axis=0) / np.sqrt(len(mat1))
            # inactive0 = np.sum(norm0 < 1e-3) / mat0.shape[1]
            # inactive1 = np.sum(norm1 < 1e-3) / mat1.shape[1]
            # logger.info("inactive ratio0 = {:.2f}%, "
            #             "inactive ratio1 = {:.2f}%"
            #             .format(100 * inactive0, 100 * inactive1))

            # logger.info(np.histogram(mat0.ravel(), bins=10))
            # logger.info(np.histogram(mat1.ravel(), bins=10))

            X = mat0
            Y = mat1
            max_epsilon = find_maximal_epsilon(X, Y)
            logger.info('max epsilon={:.3f}'.format(max_epsilon))

            tic = time.time()
            idx_X, idx_Y = find_maximal_match(X, Y, epsilon)
            toc = time.time()
            logger.info('Find max match with {:.2f}s'.format(toc - tic))

            mms = float(len(idx_X) + len(idx_Y)) / (len(X) + len(Y))
            logger.info("==> {}: max match similarity={:.2f}%".format(layer, 100 * mms))
            match_dict[layer] = {
                'idx_X': idx_X,
                'idx_Y': idx_Y,
                'similarity': mms,
                'max_epsilon': max_epsilon,
            }
        elif 'features' in layer:
            # [N, C, H, W] -> [C, N, H, W]
            mat0 = mat0.transpose([1, 0, 2, 3])
            mat1 = mat1.transpose([1, 0, 2, 3])
            mat0 = mat0.reshape([mat0.shape[0], -1])
            mat1 = mat1.reshape([mat1.shape[0], -1])
            assert mat0.shape[1] == mat1.shape[1], 'Check the sizes of two sets'

            # logger.info(np.histogram(mat0.ravel(), bins=10))
            # logger.info(np.histogram(mat1.ravel(), bins=10))

            layer_dict = {
                'idx_X': [],
                'idx_Y': [],
                'similarity': [],
                'max_epsilon': [],
            }

            tic = time.time()
            for iter in range(sample_iter):
                sample_idx = np.random.choice(mat0.shape[1], sample_ndim, replace=False)
                X = mat0[:, sample_idx]
                Y = mat1[:, sample_idx]

                idx_X, idx_Y = find_maximal_match(X, Y, epsilon)
                mms = float(len(idx_X) + len(idx_Y)) / (len(X) + len(Y))
                max_epsilon = find_maximal_epsilon(X, Y)

                layer_dict['idx_X'].append(idx_X)
                layer_dict['idx_Y'].append(idx_Y)
                layer_dict['similarity'].append(mms)
                layer_dict['max_epsilon'].append(max_epsilon)
                print('Sampling iter {}: mms = {:.2f}%, max_epsilon = {:.3f}'.format(iter, 100 * mms, max_epsilon))
            toc = time.time()

            logger.info('find max match with {:.2f}s'.format(toc - tic))
            logger.info("==> {}: max match similarity={:.2f}%".format(
                layer, 100 * np.mean(layer_dict['similarity'])))
            match_dict[layer] = layer_dict
        else:
            raise NotImplementedError()

    fname = osp.join(output_dir, 'eps_{:.3f}.pkl'.format(epsilon))
    write_pkl(match_dict, fname)
    logger.info("finish")


if __name__ == '__main__':
    main()
