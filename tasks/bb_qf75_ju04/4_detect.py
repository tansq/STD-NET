# @Time    : 2020-6-28
# @Author  : Qiushi Li
# @File    : 4_detect.py
# Image samples: 40 cover images from BossBASE
# show the excel table: conv by rank, and plot fighres


import numpy as np
import scipy.io as sio
from glob import glob
import configparser
import imageio
import argparse
import time 
#import matplotlib.pyplot as plt
from functools import partial

from setup_task_verify_experiment import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


parser = argparse.ArgumentParser("Detect_Params")
parser.add_argument('--log_root', type=str, default='../../EXP', help='log dir')
parser.add_argument('--work_name', type=str, default='detect', help='work name')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
parser.add_argument('--load_path', type=str, default='', help='model path')
args = parser.parse_args()

    
def choose_mat(layer):
    fo_mat = {
        2 : F_o_mat_l2,
        3 : F_o_mat_l3,
        4 : F_o_mat_l4,
        5 : F_o_mat_l5,
        6 : F_o_mat_l6,
        7 : F_o_mat_l7,
        8 : F_o_mat_l8,
        9 : F_o_mat_l9,
        10 : F_o_mat_l10,
        11 : F_o_mat_l11,
        12 : F_o_mat_l12,
    }
    return fo_mat.get(layer, None)

def gen_batch(cover_list, thread_idx, n_threads):

    cover_root = os.path.join(args.data_path, 'bb_256_qf75/') 

    cover_list = [cover_root + a.strip() for a in cover_list]

    load_mat=cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['im']
        img_shape = img.shape
        batch = np.empty((1,img_shape[0],img_shape[1],1), dtype='float32')
    else:
        img = imageio.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((1,img_shape[0],img_shape[1],1), dtype='uint8')
    
    labels = np.array([0], dtype='uint8')
    while True:
        for cover_path in list(cover_list):
            if load_mat:
                batch[0, :, :, 0] = sio.loadmat(cover_path)['im']
            else:
                batch[0, :, :, 0] = imageio.imread(cover_path)
            yield [batch, labels]

def init_config(config_log):
    cf = configparser.ConfigParser()
    cf.add_section('first_layer_input_ranks')
    cf.add_section('first_layer_output_ranks')
    cf.add_section('second_layer_input_ranks')
    cf.add_section('second_layer_output_ranks')
    cf.set('first_layer_input_ranks','layer2','0.0')
    cf.set('first_layer_output_ranks','layer2','0.0')
    cf.set('first_layer_input_ranks','layer3','0.0')
    cf.set('first_layer_output_ranks','layer3','0.0')
    cf.set('first_layer_input_ranks','layer4','0.0')
    cf.set('first_layer_output_ranks','layer4','0.0')
    cf.set('first_layer_input_ranks','layer5','0.0')
    cf.set('first_layer_output_ranks','layer5','0.0')
    cf.set('first_layer_input_ranks','layer6','0.0')
    cf.set('first_layer_output_ranks','layer6','0.0')
    cf.set('first_layer_input_ranks','layer7','0.0')
    cf.set('first_layer_output_ranks','layer7','0.0')
    cf.set('first_layer_input_ranks','layer8','0.0')
    cf.set('first_layer_output_ranks','layer8','0.0')
    cf.set('first_layer_input_ranks','layer9','0.0')
    cf.set('first_layer_output_ranks','layer9','0.0')
    cf.set('first_layer_input_ranks','layer10','0.0')
    cf.set('first_layer_output_ranks','layer10','0.0')
    cf.set('first_layer_input_ranks','layer11','0.0')
    cf.set('first_layer_output_ranks','layer11','0.0')
    cf.set('first_layer_input_ranks','layer12','0.0')
    cf.set('first_layer_output_ranks','layer12','0.0')

    cf.set('second_layer_input_ranks','layer3','0.0')
    cf.set('second_layer_output_ranks','layer3','0.0')
    cf.set('second_layer_input_ranks','layer4','0.0')
    cf.set('second_layer_output_ranks','layer4','0.0')
    cf.set('second_layer_input_ranks','layer5','0.0')
    cf.set('second_layer_output_ranks','layer5','0.0')
    cf.set('second_layer_input_ranks','layer6','0.0')
    cf.set('second_layer_output_ranks','layer6','0.0')
    cf.set('second_layer_input_ranks','layer7','0.0')
    cf.set('second_layer_output_ranks','layer7','0.0')
    cf.set('second_layer_input_ranks','layer8','0.0')
    cf.set('second_layer_output_ranks','layer8','0.0')
    cf.set('second_layer_input_ranks','layer9','0.0')
    cf.set('second_layer_output_ranks','layer9','0.0')
    cf.set('second_layer_input_ranks','layer10','0.0')
    cf.set('second_layer_output_ranks','layer10','0.0')
    cf.set('second_layer_input_ranks','layer11','0.0')
    cf.set('second_layer_output_ranks','layer11','0.0')
    cf.set('second_layer_input_ranks','layer12','0.0')
    cf.set('second_layer_output_ranks','layer12','0.0')

    cf.write(open(config_log, 'w'))


if __name__ == '__main__':
    # Loading data
    cover_txt_list = []
    cover_txt_list.append(BB_TRAIN_TXT)
    cover_txt_list.append(BB_VALID_TXT)
    cover_txt_list.append(BB_TEST_TXT)

    all_input_covers = []
    for txt_i in range(len(cover_txt_list)):
        with open(cover_txt_list[txt_i]) as f:
            cur_list = f.readlines()
            cur_list = [a.strip() for a in cur_list]
        all_input_covers.extend(cur_list)
    num_list = len(all_input_covers)
    pop_list = []
    print(num_list)
    # pop BOWS2 dataset
    for cover_i in range(num_list):
        if 'BOWS2' in os.path.basename(all_input_covers[cover_i]):
            pop_list.append(all_input_covers[cover_i])
    print("len(pop_list)=%d"%(len(pop_list)))
    for item in pop_list:
        all_input_covers.remove(item)
    print("Total cover: ", len(all_input_covers))
    assert len(all_input_covers) == 10000, "len(all_input_covers)=%d is not equal to 1w !!"%(len(all_input_covers))

    all_input_covers = all_input_covers[:40]

    batch_size = 40
    ds_size = 40
    start_img = 0
    end_img = 40

    # log_path = decom_log_path
    log_path = args.log_root

    output_log_dir = os.path.join(log_path, time.strftime("%Y%m%d-%H%M%S-") + args.work_name) # path for a log direcotry
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
    output_log_file = os.path.join(output_log_dir, time.strftime("%Y%m%d-") + '2_12_cand_detect_rank_40images.txt')
    if not os.path.exists(output_log_file):
        os.mknod(output_log_file)
    config_log = os.path.join(output_log_dir, 'bb_qf75ju04_v3.cfg')
    config_log_v1 = os.path.join(output_log_dir, 'bb_qf75ju04_v1.cfg')

    inspect_gen = partial(gen_batch, all_input_covers)


    output_log = open(output_log_file, 'a+')
    # index = 1  #index 1 for the first conv layer, index 2 for the second conv layer
    load_path_ori = args.load_path

    F_o_mat_l2, F_o_mat_l3, F_o_mat_l4, F_o_mat_l5, F_o_mat_l6, F_o_mat_l7, F_o_mat_l8, F_o_mat_l9, F_o_mat_l10, F_o_mat_l11, F_o_mat_l12 = \
                get_all_f_o_L2L12(snc64.SRNetC64, inspect_gen, batch_size, ds_size, load_path_ori)


    ma_threshold_bb_qf75_ju04 = [0.4, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.75, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.01]

    # get a batch images ma_mat (11, batch_size, :)

    ma_thresh = ma_threshold_bb_qf75_ju04
    err = 0.001
    row = 0

    if not os.path.exists(config_log):
        init_config(config_log)
    if not os.path.exists(config_log_v1):
        init_config(config_log_v1)
        
    for layer in range(2,13):
        for index in [1,2]:
            print(time.strftime("%Y%m%d-%H%M%S"))
            if layer == 2 and index == 2:
                continue
            model_index = layer*10+index
            mdoel_class = choose_model_class(model_index)
            fo_mat = choose_mat(layer)
            print("Processing conv: ", model_index)
            rank_results = []
            
            """
            if want to detect the first conv layer of the third layer, than change sn2.SRNet2 to sn3.SRNet3 and 
            set index as 1, remember import the SRNet_3.py in setup_task_verify_experiment.py
            """

            rank_results = inspect_decom_conv_before_cand(mdoel_class, inspect_gen, batch_size, ds_size, \
                                    load_path_ori, layer, index, output_log, fo_mat[index-1, :, :], ma_thresh, config_log_v1)
            for cand in range(1,3):
                inspect_decom_conv_cand(mdoel_class, inspect_gen, batch_size, ds_size, \
                                load_path_ori, layer, index, output_log, fo_mat[index-1, :, :], ma_thresh, rank_results, err, cand, config_log)
    output_log.close()
