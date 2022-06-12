# @Time    : 2020-6-28
# @Author  : Qiushi Li
# @File    : export_table_errorbar.py

import argparse
import pandas as pd
import scipy.io as sio
from glob import glob
import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../../libs/decom_srnet/')

from inspect_decom_snl9l12_c64 import plot_errorbar, single_plot_errorbar, plot_rank_ma_for_conv

# # start with '---'

# # start with 'rate'

parser = argparse.ArgumentParser("export_table_figure_Params")
parser.add_argument('--ma_path', type=str, default='', help='Dir of M_A.mat files')
args = parser.parse_args()

save_mat_dir = args.ma_path

# mat_path_list = glob(save_mat_dir + "ma*.mat")z

l2l7_ma_files = glob(save_mat_dir + "ma_l2l7_*.mat")
l2l7_ma_size = len(l2l7_ma_files) * 200
print("l2l7_ma_size: ", l2l7_ma_size)
l8l9conv1_ma_files = glob(save_mat_dir + "ma_l8l9conv1_*.mat")
l8l9conv1_ma_size = len(l8l9conv1_ma_files) * 200
print("l8l9conv1_ma_size: ", l8l9conv1_ma_size)
l9conv2_l10l12_ma_files = glob(save_mat_dir + "ma_l9conv2_l10l12_*.mat")
l9conv2_l10l12_ma_size = len(l9conv2_l10l12_ma_files) * 200
print("l9conv2_l10l12_ma_size: ", l9conv2_l10l12_ma_size)

# ds_size=3000
l2l7_ma_mat = np.empty((11, l2l7_ma_size, 14))
l8l9conv1_ma_mat = np.empty((3, l8l9conv1_ma_size, 14))
l9conv2_l10l12_ma_mat = np.empty((7, l9conv2_l10l12_ma_size, 59))


for i, file_path in enumerate(l2l7_ma_files):
    l2l7_ma_mat[:, i*200:(i+1)*200, :] = sio.loadmat(file_path)['ma']

for i, file_path in enumerate(l8l9conv1_ma_files):
    l8l9conv1_ma_mat[:, i*200:(i+1)*200, :] = sio.loadmat(file_path)['ma']

for i, file_path in enumerate(l9conv2_l10l12_ma_files):
    l9conv2_l10l12_ma_mat[:, i*200:(i+1)*200, :] = sio.loadmat(file_path)['ma']
    

# # compute mean and std

l2l7_mean_mat = np.mean(l2l7_ma_mat, axis=1)
l2l7_std_mat = np.std(l2l7_ma_mat, axis=1)
l8l9conv1_mean_mat = np.mean(l8l9conv1_ma_mat, axis=1)
l8l9conv1_std_mat = np.std(l8l9conv1_ma_mat, axis=1)
l9conv2_l10l12_mean_mat = np.mean(l9conv2_l10l12_ma_mat, axis=1)
l9conv2_l10l12_std_mat = np.std(l9conv2_l10l12_ma_mat, axis=1)


# row labels
convname_list=[]
for l_index in range(2,13):
    for conv_index in [1,2]:
        if l_index == 2 and conv_index == 2:
            continue
        convname_list.append('l' + str(l_index) + '-conv' + str(conv_index))
print("len(convname_list)==21: ", len(convname_list)==21)

# column labels
l2l9conv1_rank_list=[]
for i in range(0,17):
    if i in [5, 10, 15]:
        continue
    l2l9conv1_rank_list.append("%.2f"%(0.05*(20-i)))
print("len(l2l9conv1_rank_list) == 14: ", len(l2l9conv1_rank_list) == 14)
print(l2l9conv1_rank_list)

l9conv2_l10l12_rank_list=[]
for i in range(59):
    l9conv2_l10l12_rank_list.append("%d"%(64-i))
print("len(l9conv2_l10l12_rank_list) == 59: ", len(l9conv2_l10l12_rank_list) == 59)
print(l9conv2_l10l12_rank_list)

# combine mean_mat and std_mat
l2l7_mean_std = []
for i_rows in range(l2l7_mean_mat.shape[0]):
    mean_std_row = []
    for i_cols in range(l2l7_mean_mat.shape[1]):
        mean_std_row.append("(%.4f, %.4f)"%(l2l7_mean_mat[i_rows,i_cols],l2l7_std_mat[i_rows,i_cols]))    
    l2l7_mean_std.append(mean_std_row)

l8l9conv1_mean_std = []
for i_rows in range(l8l9conv1_mean_mat.shape[0]):
    mean_std_row = []
    for i_cols in range(l8l9conv1_mean_mat.shape[1]):
        mean_std_row.append("(%.4f, %.4f)"%(l8l9conv1_mean_mat[i_rows,i_cols],l8l9conv1_std_mat[i_rows,i_cols]))    
    l8l9conv1_mean_std.append(mean_std_row)

l9conv2_l10l12_mean_std = []
for i_rows in range(l9conv2_l10l12_mean_mat.shape[0]):
    mean_std_row = []
    for i_cols in range(l9conv2_l10l12_mean_mat.shape[1]):
        mean_std_row.append("(%.4f, %.4f)"%(l9conv2_l10l12_mean_mat[i_rows,i_cols],l9conv2_l10l12_std_mat[i_rows,i_cols]))    
    l9conv2_l10l12_mean_std.append(mean_std_row)

# export excel
xlsx_name = 'inspect_snc64_bbqf75ju04_%s_ma_mean.xlsx'%(str(l9conv2_l10l12_ma_size))
with pd.ExcelWriter(xlsx_name) as writer:  
    df1 = pd.DataFrame(l2l7_mean_std,
                    index=convname_list[:11],
                    columns=l2l9conv1_rank_list)
    df2 = pd.DataFrame(l8l9conv1_mean_std,
                    index=convname_list[11:14],
                    columns=l2l9conv1_rank_list)
    df3 = pd.DataFrame(l9conv2_l10l12_mean_std,
                    index=convname_list[14:],
                    columns=l9conv2_l10l12_rank_list)
    df1.to_excel(writer, sheet_name='L2L7_MA_mean_std')
    df2.to_excel(writer, sheet_name='L8L9conv1_MA_mean_std')
    df3.to_excel(writer, sheet_name='L9conv2L10L12_MA_mean_std')


# plot errorbar
spec_row=[]
spec_row = [row for row in range(len(convname_list[3:]))]
save_plot_path = "../../figs/bb_qf75ju04/imgs_%s/"%(str(l9conv2_l10l12_ma_size))
if not os.path.exists(save_plot_path):
    os.makedirs(save_plot_path)
# fig_name = "inspect_boss_ma_mean"
# plot_rank_ma_for_conv(convname_list, rank_list, mean_mat, fig_name, save_plot_path, spec_row)
# fig_name = "inspect_boss_ma_std"
# plot_rank_ma_for_conv(convname_list, rank_list, std_mat, fig_name, save_plot_path, spec_row)

fig_name = "l2l7_errorbar_mean_std"
spec_row = [row for row in range(len(convname_list[:11]))]
plot_errorbar(convname_list[:11], l2l9conv1_rank_list, l2l7_mean_mat, l2l7_std_mat, fig_name, save_plot_path, spec_row)
single_plot_errorbar(convname_list[:11], l2l9conv1_rank_list, l2l7_mean_mat, l2l7_std_mat, fig_name, save_plot_path, spec_row)

fig_name = "l8l9conv1_errorbar_mean_std"
spec_row = [row for row in range(len(convname_list[11:14]))]
plot_errorbar(convname_list[11:14], l2l9conv1_rank_list, l8l9conv1_mean_mat, l8l9conv1_std_mat, fig_name, save_plot_path, spec_row)
single_plot_errorbar(convname_list[11:14], l2l9conv1_rank_list, l8l9conv1_mean_mat, l8l9conv1_std_mat, fig_name, save_plot_path, spec_row)

fig_name = "l9conv2_l10l12_errorbar_mean_std"
spec_row = [row for row in range(len(convname_list[14:21]))]
plot_errorbar(convname_list[14:21], l9conv2_l10l12_rank_list, l9conv2_l10l12_mean_mat, l9conv2_l10l12_std_mat, fig_name, save_plot_path, spec_row)
single_plot_errorbar(convname_list[14:21], l9conv2_l10l12_rank_list, l9conv2_l10l12_mean_mat, l9conv2_l10l12_std_mat, fig_name, save_plot_path, spec_row)
# plot_rank_ma_for_conv(convname_list, rank_list, mean_mat, fig_name, save_plot_path, spec_row)
