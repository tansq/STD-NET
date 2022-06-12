import os
import sys
import time
sys.path.append('../../libs/decom_srnet/')
from models import *

# from decom_srnet import copy_weights_to_decom_srnet
from detect_srnet_before_cand import *
from detect_srnet_cand import *

from inspect_decom_snl9l12_c64 import inspect_decom_conv, get_all_f_o_L2L12


#BOSSBase + BOWS2 dataset path

BB_TRAIN_TXT = '../../datas/bb/jpeg/train.txt'
BB_VALID_TXT = '../../datas/bb/jpeg/valid.txt'
BB_TEST_TXT = '../../datas/bb/jpeg/test.txt'


#training config
# for bb dataset, totaly 40000 images
train_batch_size = 32
valid_batch_size = 40
test_batch_size = 40
max_iter = 500000
train_interval = 875    # 875 iters/epoch for bb
valid_interval = 875
test_interval = 875
save_interval = 875
num_runner_threads = 10
optimizer = snc64.AdamaxOptimizer
learning_rate = [0.001, 0.0001]
learning_rate_boundaries = [400000]

ori_log_path = '../../log_dir/'

# fine-tune
transfer_max_iter = 150000
transfer_learning_rate = [0.001, 0.0001]
transfer_learning_rate_boundaries = [100000]

transfer_max_iter = 250000
transfer_learning_rate = [0.001, 0.0001, 0.00001]
transfer_learning_rate_boundaries = [100000, 200000]
