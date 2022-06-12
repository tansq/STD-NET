# @Time    : 2020-6-28
# @Author  : Qiushi Li
# @File    : 5_decom.py

import os
import sys
import argparse
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7' # set a GPU (with GPU Number)
    
from setup_task_verify_experiment import *


parser = argparse.ArgumentParser("Initial_decom_Params")
parser.add_argument('--log_root', type=str, default='../../EXP', help='log dir')
parser.add_argument('--work_name', type=str, default='init_decom', help='work name')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
parser.add_argument('--load_path', type=str, default='', help='model to be decomposed')
parser.add_argument('--load_config', type=str, default='', help='config path')
args = parser.parse_args()

assert args.load_path != '', "Please specify args.load_path !"
assert args.load_config != '', "Please specify args.load_config !"

if __name__ == '__main__':
    TRAIN_TXT = BB_TRAIN_TXT
    VALID_TXT = BB_VALID_TXT

    COVER_ROOT = os.path.join(args.data_path, 'bb_256_qf75/')
    STEGO_ROOT = os.path.join(args.data_path, 'bb_256_qf75_juniward04/')
    train_gen = snc64.partial(snc64.gen_flip_and_rot, \
                        COVER_ROOT, STEGO_ROOT, \
                            TRAIN_TXT, TRAIN_TXT) 
    valid_gen = snc64.partial(snc64.gen_valid, \
                        COVER_ROOT, STEGO_ROOT, \
                            VALID_TXT, VALID_TXT)


    # load_path = None                       
    save_path = os.path.join(args.log_root, time.strftime("%Y%m%d-%H%M%S-") + args.work_name) # path for a log direcotry


    with open(TRAIN_TXT) as f:
        train_cover = f.readlines()
        train_cover_list = [a.strip() for a in train_cover]

    with open(VALID_TXT) as f:
        val_cover = f.readlines()
        val_cover_list = [a.strip() for a in val_cover]

    train_ds_size = len(train_cover_list) * 2
    valid_ds_size = len(val_cover_list) * 2
    print ('train_ds_size: %i'%train_ds_size)
    print ('valid_ds_size: %i'%valid_ds_size)

    if valid_ds_size % valid_batch_size != 0:
        raise ValueError("change batch size for validation")
        
    # load_path_ori = args.load_path
    boundaries = learning_rate_boundaries     # learning rate adjustment at iteration 400K
    values = learning_rate  # learning rates

    decom_model_name = 'model_decom_finetune.ckpt'
    model_name = 'model_decom_init.ckpt'
    snd.initial_and_decom_model(snd.SRNetDecom, train_gen, valid_gen , train_batch_size, valid_batch_size, valid_ds_size, \
        optimizer, boundaries, values, train_interval, valid_interval, \
        save_path, model_name, args.load_path, args.load_config, decom_model_name, num_runner_threads)
