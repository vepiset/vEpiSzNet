import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()

config.TRAIN.process_num = 4

config.TRAIN.batch_size = 128
config.TRAIN.validatiojn_batch_size = config.TRAIN.batch_size
config.TRAIN.accumulation_batch_size = 128
config.TRAIN.log_interval = 10
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 10

config.TRAIN.init_lr = 0.0005
config.TRAIN.lr_scheduler = 'cos'


if config.TRAIN.lr_scheduler == 'ReduceLROnPlateau':
    config.TRAIN.epoch = 100
    config.TRAIN.lr_scheduler_factor = 0.1

config.TRAIN.weight_decay_factor = 5.e-4
config.TRAIN.warmup_step = 1000
config.TRAIN.opt = 'Adamw'
config.TRAIN.gradient_clip = 1

config.TRAIN.mix_precision = True

config.MODEL = edict()

config.MODEL.model_path = './models/'

config.DATA = edict()

config.DATA.data_file = 'data.csv'

config.DATA.data_root_path = '.'

#no early stop
config.MODEL.early_stop = 100000

config.MODEL.pretrained_model = None

config.SEED = 42

from lib.utils.seed_utils import seed_everything

seed_everything(config.SEED)
