# @TIME: 2020-6-28
# @AUTHOR: Qiushi Li
# @Function: Traverse and compute M_A
# Image samples: all cover images from specified dataset

import numpy as np
import scipy.io as sio
from setup_task_verify_experiment import *
from glob import glob
# import configparser
import argparse

import time 
#import matplotlib.pyplot as plt
from functools import partial
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
parser = argparse.ArgumentParser("Inspect_Params")
parser.add_argument('--log_root', type=str, default='', help='the log path root')
parser.add_argument('--work_name', type=str, default='', help='the log file saved dir')
parser.add_argument('--load_path', type=str, default='', help='model loading path')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
args = parser.parse_args()

def gen_batch(cover_list, thread_idx, n_threads):

    img = sio.loadmat(cover_list[0])['im']
    img_shape = img.shape
    batch = np.empty((1, img_shape[0], img_shape[1], 1), dtype='float32')
    
    labels = np.array([0], dtype='uint8')
    while True:
        for cover_path in list(cover_list):
            batch[0, :, :, 0] = sio.loadmat(cover_path)['im']
            yield [batch, labels]

load_path_ori = args.load_path

# Loading datas
cover_dir = os.path.join(args.data_path, 'bb_256_qf75/')
stego_dir = os.path.join(args.data_path, 'bb_256_qf75_juniward04/')
all_input_imgs = sorted(glob(cover_dir + '/BOWS2_*.mat'))
# all_input_stegos = sorted(glob(stego_dir + '/BOWS2_*.mat'))
# all_input_imgs = all_input_imgs + all_input_stegos
# num_list = len(all_input_covers)
random.seed(2021)
random.shuffle(all_input_imgs)
all_input_imgs = all_input_imgs[:10000]
# assert num_list == 10000, "len(all_input_covers)=%d is not equal to 10k !!"%(num_list)


# log_path = ori_log_path
log_path = args.log_root

if args.work_name != '':
    output_log_dir = os.path.join(log_path, time.strftime("%Y%m%d-") + args.work_name) # path for a log direcotry
else:
    output_log_dir = os.path.join(log_path, time.strftime("%Y%m%d-%H%M%S-") + 'inspect')
if not os.path.exists(output_log_dir):
    os.makedirs(output_log_dir)

output_log_file = os.path.join(output_log_dir, "logs")
if not os.path.exists(output_log_file):
    os.makedirs(output_log_file)

save_mat_dir = os.path.join(output_log_dir, 'ma_mat')
if not os.path.exists(save_mat_dir):
    os.makedirs(save_mat_dir)
    
batch_size = 200
ds_size = 200
start_im = 0
end_im = 10000
output_log_file = os.path.join(output_log_file, time.strftime("%Y%m%d-%H%M%S-") + "inspect_BB_qf75_ju04_%d-%d.txt"%(start_im, end_im))
output_log = open(output_log_file, 'a+')
# ds_size = 10000  # total BOSSBASE cover images
for i_part in range(start_im, end_im, ds_size):

    input_covers = all_input_imgs[i_part:i_part+ds_size]
    
    inspect_gen = partial(gen_batch, input_covers)

    F_o_mat_l2, F_o_mat_l3, F_o_mat_l4, F_o_mat_l5, F_o_mat_l6, F_o_mat_l7, \
        F_o_mat_l8, F_o_mat_l9, F_o_mat_l10, F_o_mat_l11, F_o_mat_l12 = \
        get_all_f_o_L2L12(snc64.SRNetC64, inspect_gen, batch_size, ds_size, load_path_ori)
    
    # ----------------compute M_A---------------------------------------------

    fo_assign = {
            '2':  lambda conv_i: F_o_mat_l2[conv_i,:,:],
            '3':  lambda conv_i: F_o_mat_l3[conv_i,:,:],
            '4':  lambda conv_i: F_o_mat_l4[conv_i,:,:],
            '5':  lambda conv_i: F_o_mat_l5[conv_i,:,:],
            '6':  lambda conv_i: F_o_mat_l6[conv_i,:,:],
            '7':  lambda conv_i: F_o_mat_l7[conv_i,:,:],
            '8':  lambda conv_i: F_o_mat_l8[conv_i,:,:],
            '9':  lambda conv_i: F_o_mat_l9[conv_i,:,:],
            '10': lambda conv_i: F_o_mat_l10[conv_i,:,:],
            '11': lambda conv_i: F_o_mat_l11[conv_i,:,:],
            '12': lambda conv_i: F_o_mat_l12[conv_i,:,:]
    }
    

    l2l7_ma_mat = np.empty((11, ds_size, 14))
    l8l9conv1_ma_mat = np.empty((3, ds_size, 14))
    l9conv2_l10l12_ma_mat = np.empty((7, ds_size, 59))
    
    l8l9conv1_row = 0
    l9conv2_l10l12_row = 0
    l2l7_row = 0
    for layer in range(2,13):
        for index in [1,2]:
            
            if layer == 2 and index == 2:
                continue
            model_index = layer*10+index
            model_class = choose_model_class(model_index)
            
            ma_row = inspect_decom_conv(model_class, inspect_gen, batch_size, ds_size, \
                                    load_path_ori, layer, index, output_log, fo_assign[str(layer)](index-1))
            
            if layer == 8 or (layer == 9 and index == 1):
                l8l9conv1_ma_mat[l8l9conv1_row, :, :] = ma_row
                l8l9conv1_row = l8l9conv1_row + 1
                if l8l9conv1_row == 3:
                    save_l8l9conv1_ma_path = os.path.join(save_mat_dir, "ma_l8l9conv1_%d_%d.mat"%(i_part, i_part+ds_size))
                    sio.savemat(save_l8l9conv1_ma_path, {'ma': l8l9conv1_ma_mat})
            elif (layer <= 12 and layer >= 10) or (layer == 9 and index == 2):
                l9conv2_l10l12_ma_mat[l9conv2_l10l12_row, :, :] = ma_row
                l9conv2_l10l12_row = l9conv2_l10l12_row + 1
                if l9conv2_l10l12_row == 7:
                    save_l9conv2_l10l12_ma_path = os.path.join(save_mat_dir, "ma_l9conv2_l10l12_%d_%d.mat"%(i_part, i_part+ds_size))
                    sio.savemat(save_l9conv2_l10l12_ma_path, {'ma': l9conv2_l10l12_ma_mat})
            else:
                l2l7_ma_mat[l2l7_row, :, :] = ma_row
                l2l7_row = l2l7_row + 1
                if l2l7_row == 11:
                    save_l2l7_ma_path = os.path.join(save_mat_dir, "ma_l2l7_%d_%d.mat"%(i_part, i_part+ds_size))
                    sio.savemat(save_l2l7_ma_path, {'ma': l2l7_ma_mat})

output_log.close()
