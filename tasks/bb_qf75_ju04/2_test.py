# @Time    : 2020-6-28
# @Author  : Qiushi Li
# @File    : 2_test.py

import os
import sys
import argparse
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4' # set a GPU (with GPU Number)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from setup_task_verify_experiment import *


parser = argparse.ArgumentParser("Test_Params")
parser.add_argument('--work_name', type=str, default='', help='current work name')
parser.add_argument('--load_path', type=str, default='', help='model loading path')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
args = parser.parse_args()

TEST_TXT = BB_TEST_TXT

COVER_ROOT = os.path.join(args.data_path, 'bb_256_qf75/')
STEGO_ROOT = os.path.join(args.data_path, 'bb_256_qf75_juniward04/')
    
test_gen = snc64.partial(snc64.gen_valid, \
                    COVER_ROOT, STEGO_ROOT, \
                        TEST_TXT, TEST_TXT)

with open(TEST_TXT) as f:
    test_cover = f.readlines()
    test_cover_list = [a.strip() for a in test_cover]

LOG_DIR = ori_log_path
LOAD_CKPT = args.load_path        # loading from a specific checkpoint

if args.work_name != '':
    LOG_DIR = os.path.join(ori_log_path, args.work_name) # path for a log and model saving direcotry
else:
    LOG_DIR = os.path.join(ori_log_path, time.strftime("%Y%m%d-%H%M%S-") + 'test')

ori_acc_log_path = os.path.join(LOG_DIR, 'logs')
if not os.path.exists(ori_acc_log_path):
    os.makedirs(ori_acc_log_path)
ori_acc_log_path = os.path.join(ori_acc_log_path, 'test_acc.txt')


test_ds_size = len(test_cover_list) * 2
print ('test_ds_size: %i'%test_ds_size)

if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")

snc64.test_dataset(snc64.SRNetC64, test_gen, test_batch_size, test_ds_size, LOAD_CKPT, ori_acc_log_path)
