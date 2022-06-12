# @Time    : 2020-6-28
# @Author  : Qiushi Li
# @File    : 8_test_decom.py

import os
import sys
import argparse
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from setup_task_verify_experiment import *

parser = argparse.ArgumentParser("Test_decom_Params")
parser.add_argument('--log_root', type=str, default='../../EXP', help='log dir')
parser.add_argument('--work_name', type=str, default='test_decom', help='work name, 6_finetune or 7_train_from_scratch')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
parser.add_argument('--load_path', type=str, default='', help='model to be loaded')
parser.add_argument('--load_config', type=str, default='', help='config file path')
args = parser.parse_args()

assert args.load_path != '', "Please specify args.load_path !"
assert args.load_config != '', "Please specify args.load_config !"

if __name__ == '__main__':
    TEST_TXT = BB_TEST_TXT
    COVER_ROOT = os.path.join(args.data_path, 'bb_256_qf75/')
    STEGO_ROOT = os.path.join(args.data_path, 'bb_256_qf75_juniward04/')
    test_gen = snc64.partial(snc64.gen_valid, \
                        COVER_ROOT, STEGO_ROOT, \
                            TEST_TXT, TEST_TXT)

    LOG_DIR = os.path.join(args.log_root, time.strftime("%Y%m%d-%H%M%S-") + args.work_name) # path for a log direcotry

    LOAD_CKPT = args.load_path

    decom_acc_log_path = os.path.join(LOG_DIR, 'logs')
    if not os.path.exists(decom_acc_log_path):
        os.makedirs(decom_acc_log_path)
    decom_acc_log_path = os.path.join(decom_acc_log_path, 'test_decom_acc.txt')

    with open(TEST_TXT) as f:
        test_cover = f.readlines()
        test_cover_list = [a.strip() for a in test_cover]
    test_ds_size = len(test_cover_list) * 2
    print ('test_ds_size: %i'%test_ds_size)

    if test_ds_size % test_batch_size != 0:
        raise ValueError("change batch size for testing!")

    snd.test_dataset(snd.SRNetDecom, test_gen, test_batch_size, test_ds_size, LOAD_CKPT, decom_acc_log_path, args.load_config)
