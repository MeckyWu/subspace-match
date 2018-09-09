from .. import AttrDict

__C = AttrDict()
__C.model = ''

# train
__C.TRAIN = AttrDict()
__C.TRAIN.rng_seed = 0
__C.TRAIN.lr = 0.1
__C.TRAIN.lr_step = (80, 120)
__C.TRAIN.momentum = 0.9
__C.TRAIN.wd = 5e-4
__C.TRAIN.batch_size = 128
__C.TRAIN.end_epoch = 160

# test
__C.TEST = AttrDict()
__C.TEST.batch_size = 100