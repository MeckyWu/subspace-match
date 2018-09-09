import os
import os.path as osp
import pickle


def mkdir(dir_name):
    if not osp.exists(dir_name):
        os.makedirs(dir_name)
        print('make new directory at %s' % dir_name)
    # assert osp.exists(dir_name), '{} does not exist'.format(dir_name)
    return dir_name


def write_pkl(obj, fname):
    with open(fname, "wb") as fid:
        pickle.dump(obj, fid, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(fname):
    with open(fname, "rb") as fid:
        obj = pickle.load(fid)
    return obj


def create_logger(output_dir, log_file='', enable_console=False):
    """
    Create a logging instance
    Args:
        output_dir: directory containing log file
        log_file: prefix of log file.
        enable_console: output to console

    Returns:

    """
    import logging
    import time

    if log_file:
        log_file = log_file + '_'
    log_file += time.strftime('%Y-%m-%d-%H-%M-%S')
    log_file += '.log'

    head = '%(asctime)s %(message)s'
    datefmt = '%m-%d %H:%M:%S'
    logging.basicConfig(filename=osp.join(output_dir, log_file), format=head, datefmt=datefmt)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if enable_console:
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

    return logger
