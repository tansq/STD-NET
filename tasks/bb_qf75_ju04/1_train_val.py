import os
import sys
import argparse
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # set a GPU (with GPU Number)

from setup_task_verify_experiment import *


parser = argparse.ArgumentParser("Train_Params")
parser.add_argument('--log_root', type=str, default='', help='current log root path')
parser.add_argument('--work_name', type=str, default='', help='current work name')
parser.add_argument('--load_path', type=str, default='', help='model loading path')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
args = parser.parse_args()

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
                        
with open(TRAIN_TXT) as f:
    train_cover = f.readlines()
    train_cover_list = [a.strip() for a in train_cover]

with open(VALID_TXT) as f:
    val_cover = f.readlines()
    val_cover_list = [a.strip() for a in val_cover]

# log_path = ori_log_path
log_path = args.log_root
if args.work_name != '':
    args.work_name = time.strftime("%Y%m%d-") + args.work_name
    LOG_DIR = os.path.join(log_path, args.work_name) # path for a log direcotry
else:
    LOG_DIR = os.path.join(log_path, time.strftime("%Y%m%d-%H%M%S-") + 'train')

if args.load_path == '':
    load_path = None
else:
    load_path = args.load_path

ori_acc_log_path = os.path.join(LOG_DIR, 'logs')
if not os.path.exists(ori_acc_log_path):
    os.makedirs(ori_acc_log_path)
ori_acc_log_path = os.path.join(ori_acc_log_path, 'train_val_acc.txt')

train_ds_size = len(train_cover_list) * 2
valid_ds_size = len(val_cover_list) * 2
print ('train_ds_size: %i'%train_ds_size)
print ('valid_ds_size: %i'%valid_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")
    

boundaries = learning_rate_boundaries     # learning rate adjustment at iteration 400K
values = learning_rate  # learning rates
snc64.train(snc64.SRNetC64, train_gen, valid_gen , train_batch_size, valid_batch_size, valid_ds_size, \
      optimizer, boundaries, values, train_interval, valid_interval, max_iter,\
      save_interval, LOG_DIR, ori_acc_log_path, num_runner_threads, load_path)
