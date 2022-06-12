# 
# by liqs, 2020-6-19

import sys
import os
import tensorflow as tf
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
# from SRNetL9L12C64_2 import *
from utils import *
import time


# load batch_size=20 ds_size=20 images and get the feature map F_o of all conv
# return F_o_mat (num_conv=11, ds_size, f_o)
def get_all_f_o_L2L12(model_class, gen, batch_size, ds_size, load_path_ori):

    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NHWC')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss', \
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy', \
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000) 
    with tf.Session() as sess:
        sess.run(init_op)
        if load_path_ori is not None:
            saver.restore(sess, load_path_ori)
        print("load ckpt: ", load_path_ori)
        runner.start_threads(sess, 1)

        F_o_mat_l2 = np.zeros((1, ds_size, 256*256*16))
        F_o_mat_l3 = np.zeros((2, ds_size, 256*256*16))
        F_o_mat_l4 = np.zeros((2, ds_size, 256*256*16))
        F_o_mat_l5 = np.zeros((2, ds_size, 256*256*16))
        F_o_mat_l6 = np.zeros((2, ds_size, 256*256*16))
        F_o_mat_l7 = np.zeros((2, ds_size, 256*256*16))
        F_o_mat_l8 = np.zeros((2, ds_size, 256*256*16))
        F_o_mat_l9 = np.zeros((2, ds_size, 128*128*64))
        F_o_mat_l10 = np.zeros((2, ds_size, 64*64*64))
        F_o_mat_l11 = np.zeros((2, ds_size, 32*32*64))
        F_o_mat_l12 = np.zeros((2, ds_size, 16*16*64))

        for j in range(0, ds_size, batch_size):
            
            sess.run(increment_op)  # ds_size should be equal to batch_size
            
            row = 0
            for layer in range(2,13):
                
                if (layer >= 3 and layer <= 7) or layer == 12:
                    tensor_name_1 = 'Layer' + str(layer) + '/BatchNorm/FusedBatchNorm:0'
                    tensor_name_2 = 'Layer' + str(layer) + '/BatchNorm_1/FusedBatchNorm:0'
                elif layer >= 8 and layer <= 11:
                    tensor_name_1 = 'Layer' + str(layer) + '/BatchNorm_1/FusedBatchNorm:0'
                    tensor_name_2 = 'Layer' + str(layer) + '/BatchNorm_2/FusedBatchNorm:0'
                    # tensor_name_1 = 'Layer' + str(layer) + '/BatchNorm_1/cond_1/Merge:0'
                    # tensor_name_2 = 'Layer' + str(layer) + '/BatchNorm_2/cond_1/Merge:0'
                elif layer == 2:
                    tensor_name_1 = 'Layer' + str(layer) + '/BatchNorm/FusedBatchNorm:0'
                    ori_bn1 = sess.graph.get_tensor_by_name(tensor_name_1)
                    bn1_f = sess.run(ori_bn1)
                    bn1_f = bn1_f.reshape(batch_size,-1)
                    row = row + 1
                    F_o_mat_l2[0, j:j+batch_size, :] = bn1_f
                    continue
                    
                    
                ori_bn1 = sess.graph.get_tensor_by_name(tensor_name_1)
                bn1_f = sess.run(ori_bn1)
                bn1_f = bn1_f.reshape(batch_size,-1)
                # fo_assign[str(layer)](0, j, j+batch_size, bn1_f)
                row = row + 1

                ori_bn2 = sess.graph.get_tensor_by_name(tensor_name_2)
                bn2_f = sess.run(ori_bn2)
                bn2_f = bn2_f.reshape(batch_size,-1)
                # fo_assign[str(layer)](1, j, j+batch_size, bn2_f)   
                row = row + 1

                if layer == 3:
                    F_o_mat_l3[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l3[1, j:j+batch_size, :] = bn2_f
                elif layer == 4:
                    F_o_mat_l4[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l4[1, j:j+batch_size, :] = bn2_f
                elif layer == 5:
                    F_o_mat_l5[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l5[1, j:j+batch_size, :] = bn2_f
                elif layer == 6:
                    F_o_mat_l6[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l6[1, j:j+batch_size, :] = bn2_f
                elif layer == 7:
                    F_o_mat_l7[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l7[1, j:j+batch_size, :] = bn2_f
                elif layer == 8:
                    F_o_mat_l8[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l8[1, j:j+batch_size, :] = bn2_f
                elif layer == 9:
                    F_o_mat_l9[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l9[1, j:j+batch_size, :] = bn2_f
                elif layer == 10:
                    F_o_mat_l10[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l10[1, j:j+batch_size, :] = bn2_f
                elif layer == 11:
                    F_o_mat_l11[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l11[1, j:j+batch_size, :] = bn2_f
                elif layer == 12:
                    F_o_mat_l12[0, j:j+batch_size, :] = bn1_f
                    F_o_mat_l12[1, j:j+batch_size, :] = bn2_f
                            
        mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                             accuracy_summary.mean_variable])
        
        print("row==21: ", (row==21))
        assert row==21, "row is not equal to 21, warning ! "

    return F_o_mat_l2, F_o_mat_l3, F_o_mat_l4, F_o_mat_l5, F_o_mat_l6, F_o_mat_l7, \
        F_o_mat_l8, F_o_mat_l9, F_o_mat_l10, F_o_mat_l11, F_o_mat_l12


"""
traverse shrinking rate from 1 to 0.2
the matrix rank is reduced by 5% every time
print: the M_A after each reduction
return: M_A_list
"""
def inspect_decom_conv(model_class, gen, batch_size, ds_size, \
                         load_path_ori, layer, index, output_log, F_o_mat):
    
    # ma_row = np.empty((ds_size, 59))
    mat_col = 0
    print("Ready to traverse shrinking rate in l%d-conv%d" % (layer, index))
    output_log.write('--------layer' + str(layer) + '--------' + time.strftime("%Y%m%d-%H%M%S") + '\n')
    output_log.write('------' + str(index) + '------' + '\n') 
    
    var_name, var_name_bias, fc_tensor_name = get_decom_tensor_name(layer, index)
    var_ori_1 = tf.contrib.framework.load_variable(load_path_ori, var_name)
    var_bias_1 = tf.contrib.framework.load_variable(load_path_ori, var_name_bias)

    if (layer >= 2 and layer <= 8) or (layer == 9 and index == 1):
        ma_dim = 14
        ma_row = np.empty((ds_size, ma_dim))
        
        # if index == 1:
        step = 0.05
        for i in range(0,17):   # rate reduction from 1 to 0.20 
            if i in [5, 10, 15]:
                continue
            tf.reset_default_graph()
            runner = GeneratorRunner(gen, batch_size * 10)
            img_batch, label_batch = runner.get_batched_inputs(batch_size)
            model = model_class(False, 'NHWC')
            model._build_model(img_batch)
            loss, accuracy = model._build_losses(label_batch)
            loss_summary = average_summary(loss, 'loss', \
                                        float(ds_size) / float(batch_size))
            accuracy_summary = average_summary(accuracy, 'accuracy', \
                                            float(ds_size) / float(batch_size))
            increment_op = tf.group(loss_summary.increment_op, \
                                    accuracy_summary.increment_op)
            # global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
            #                               initializer=tf.constant_initializer(0), \
            #                               trainable=False)
            init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            saver = tf.train.Saver(max_to_keep=10000) 
            with tf.Session() as sess:
                sess.run(init_op)
                runner.start_threads(sess, 1)
                ranks = []
                ranks = [int(var_ori_1.shape[2] * step * (20-i)), int(var_ori_1.shape[3] * step * (20-i))] 
                core_1, [first_1, last_1] = partial_tucker(var_ori_1,modes=[2,3],ranks=ranks,init='svd')
                first_1 = np.float32(np.expand_dims(np.expand_dims(first_1.copy(), axis=0),axis=0))
                last_1 = last_1.transpose((1,0))
                last_1 = np.float32(np.expand_dims(np.expand_dims(last_1.copy(), axis=0),axis=0))
                
                assign_sess(sess, load_path_ori, layer, index, first_1, core_1, last_1, var_bias_1)
               
                
                for j in range(0, ds_size, batch_size):
                    sess.run(increment_op)
                    # load the decom_conv+bn from decom_srnet_layeri and get the feature map
                    decom_bn = sess.graph.get_tensor_by_name(fc_tensor_name)
                    F_c = sess.run(decom_bn)

                    F_c = F_c.reshape(batch_size,-1)
                    # F_o = F_o_mat[batch_size,:]

                    print("F_c.shape: ", F_c.shape)
                    M_A, _ = compute_ma(batch_size, F_o_mat[j:j+batch_size,:], F_c)  
                    ma_row[j:j+batch_size,mat_col] = M_A
                    
                mat_col = mat_col + 1
                mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                         accuracy_summary.mean_variable])
                print("Rank: {}, time: {}".format(ranks,  time.strftime("%Y%m%d-%H%M%S")))
                output_log.write("Rank: {}, acc={}, loss={}, time: {} \n".format(ranks, \
                    str(mean_accuracy), str(mean_loss), time.strftime("%Y%m%d-%H%M%S")))
                output_log.flush()
                sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])

    elif (layer >= 10 and layer <= 12) or (layer == 9 and index == 2):
    # elif layer >= 10 and layer <= 11:
    # elif layer >= 8 and layer <= 11:
        ma_dim = 59
        ma_row = np.empty((ds_size, ma_dim))
        
        # if index == 1:
        # step = 0.05
        for i in range(ma_dim):
            tf.reset_default_graph()
            runner = GeneratorRunner(gen, batch_size * 10)
            img_batch, label_batch = runner.get_batched_inputs(batch_size)
            model = model_class(False, 'NHWC')
            model._build_model(img_batch)
            loss, accuracy = model._build_losses(label_batch)
            loss_summary = average_summary(loss, 'loss', \
                                        float(ds_size) / float(batch_size))
            accuracy_summary = average_summary(accuracy, 'accuracy', \
                                            float(ds_size) / float(batch_size))
            increment_op = tf.group(loss_summary.increment_op, \
                                    accuracy_summary.increment_op)
            # global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
            #                               initializer=tf.constant_initializer(0), \
            #                               trainable=False)
            init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            saver = tf.train.Saver(max_to_keep=10000) 
            with tf.Session() as sess:
                sess.run(init_op)
                runner.start_threads(sess, 1)
                ranks = []
                ranks = [int(var_ori_1.shape[2] - i), int(var_ori_1.shape[3] - i)]
                core_1, [first_1, last_1] = partial_tucker(var_ori_1,modes=[2,3],ranks=ranks,init='svd')
                first_1 = np.float32(np.expand_dims(np.expand_dims(first_1.copy(), axis=0),axis=0))
                last_1 = last_1.transpose((1,0))
                last_1 = np.float32(np.expand_dims(np.expand_dims(last_1.copy(), axis=0),axis=0))
                
                assign_sess(sess, load_path_ori, layer, index, first_1, core_1, last_1, var_bias_1)
                               
                for j in range(0, ds_size, batch_size):
                    sess.run(increment_op)
                    decom_bn = sess.graph.get_tensor_by_name(fc_tensor_name)
                    F_c = sess.run(decom_bn)

                    F_c = F_c.reshape(batch_size,-1)
                    # F_o = F_o_mat[batch_size,:]

                    print("F_c.shape: ", F_c.shape)                        
                    M_A, _ = compute_ma(batch_size, F_o_mat[j:j+batch_size,:], F_c)
                    ma_row[j:j+batch_size,mat_col] = M_A
                    
                mat_col = mat_col + 1
                mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable,
                                                     accuracy_summary.mean_variable])
                print("Rank: {}, time: {}".format(ranks,  time.strftime("%Y%m%d-%H%M%S")))
                output_log.write("Rank: {}, acc={}, loss={}, time: {} \n".format(ranks, \
                    str(mean_accuracy), str(mean_loss), time.strftime("%Y%m%d-%H%M%S")))
                # output_log.write('rate ' + str(round((20-i)*step, 2)) + ',acc ' + str(mean_accuracy) + ',loss ' + str(
                #     mean_loss) + '\n')
                output_log.flush()
                sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
    
    return ma_row
            
import matplotlib.pyplot as plt

# plot a curve graph for every conv of boss images
def plot_errorbar(conv_list, rank_list, mean_mat, std_err, fig_name, save_plot_path, spec_row):
    # spec_row (list) : specify rows to plot
    conv_array = np.array(conv_list)
    rank_array = np.array(rank_list)

    conv_array = conv_array[spec_row]
    # ma_mat = ma_mat[spec_row,:]
    mean_mat = mean_mat[spec_row,:]
    std_err = std_err[spec_row,:]
    
    # Common figsizes: (10, 7.5) and (12, 9)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary here.
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    # plt.xticks(range(len(rank_list)),rank_list, fontsize=10)
    # plt.yticks(fontsize=10)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    # ax.grid(True, 'major', 'y', ls='-', lw=.5, c='k', alpha=.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just
    # plotted.
    ax.tick_params(axis='both', which='both', bottom=True, top=False,
                    labelbottom=True, left=True, right=False, labelleft=True)
    # plt.figure(figsize=(10,5))# set fig canvas size
    fig.suptitle(fig_name,fontsize=17,ha="center")
    plt.ylim(-0.05, 1.35)

    for i in range(len(spec_row)):
        ax.errorbar(x=[k for k in range(len(rank_list))], y=mean_mat[i,:], yerr=std_err[i,:], label=conv_array[i], elinewidth=1,capsize=3)
        # ax.plot(mean_mat[i,:],linewidth=2,linestyle='-',label=conv_array[i], marker='o')
        # plt.plot(rank_array, ma_mat[1,:],color="darkblue",linewidth=1,linestyle='--', marker='o')
        # plt.plot(rank_array, ma_mat[2,:],color="goldenrod",linewidth=1.5,linestyle='-', marker='o')
    
    if 'l10' in conv_array[i] or 'l11' in conv_array[i] or 'l12' in conv_array[i] or 'l9-conv2' in conv_array[i]:
        plt.xticks(range(0, len(rank_list), 10), rank_list[::10], fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(u'rank',fontsize=14)
        plt.ylabel(u'M_A',fontsize=14)
    else:
        plt.xticks(range(len(rank_list)), rank_list, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(u'rate',fontsize=14)
        plt.ylabel(u'M_A',fontsize=14)
    
    plt.legend(bbox_to_anchor=(0.5, -0.06), loc="upper center", ncol=6, frameon=False)
    plt.savefig(save_plot_path + fig_name + ".png", bbox_inches='tight', dpi=100)
    plt.close(fig_name)


# plot a curve graph for every conv of boss images
def single_plot_errorbar(conv_list, rank_list, mean_mat, std_err, fig_name, save_plot_path, spec_row):
    # spec_row (list) : specify rows to plot
    conv_array = np.array(conv_list)
    rank_array = np.array(rank_list)

    conv_array = conv_array[spec_row]
    # ma_mat = ma_mat[spec_row,:]
    mean_mat = mean_mat[spec_row,:]
    std_err = std_err[spec_row,:]

    for i in range(len(spec_row)):
        # Common figsizes: (10, 7.5) and (12, 9)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        # Ensure that the axis ticks only show up on the bottom and left of the plot.
        # Ticks on the right and top of the plot are generally unnecessary.
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


        # Remove the tick marks; they are unnecessary with the tick lines we just
        # plotted.
        ax.tick_params(axis='both', which='both', bottom=True, top=False,
                        labelbottom=True, left=True, right=False, labelleft=True)
        # plt.figure(figsize=(10,5))# set fig canvas size
        fig.suptitle(conv_array[i],fontsize=17,ha="center")
        plt.ylim(-0.05, 1.35)
        # if 'l9' in conv_array[i]:
            
        ax.errorbar(x=[k for k in range(len(rank_list))], y=mean_mat[i,:], yerr=std_err[i,:], label=conv_array[i], elinewidth=1,capsize=3)

        if 'l10' in conv_array[i] or 'l11' in conv_array[i] or 'l12' in conv_array[i] or 'l9-conv2' in conv_array[i]:
            plt.xticks(range(0, len(rank_list), 10), rank_list[::10], fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel(u'rank',fontsize=14)
            plt.ylabel(u'M_A',fontsize=14)
        else:
            plt.xticks(range(len(rank_list)), rank_list, fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel(u'rate',fontsize=14)
            plt.ylabel(u'M_A',fontsize=14)

        # x_major_locator=plt.MultipleLocator(10)
        # ax.xaxis.set_major_locator(x_major_locator)

        # plt.legend(bbox_to_anchor=(0.5, -0.06), loc="upper center", ncol=6, frameon=False)
        plt.savefig(save_plot_path + conv_array[i] + ".png", bbox_inches='tight', dpi=100)

# plot a curve graph for every conv of boss images
def plot_rank_ma_for_conv(conv_list, rank_list, ma_mat, fig_name, save_plot_path, spec_row):
    # spec_row (list) : specify rows to plot
    conv_array = np.array(conv_list)
    rank_array = np.array(rank_list)

    conv_array = conv_array[spec_row]
    ma_mat = ma_mat[spec_row,:]
    
    # Common figsizes: (10, 7.5) and (12, 9)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary here.
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(range(len(rank_list)),rank_list, fontsize=10)
    plt.yticks(fontsize=10)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    ax.grid(True, 'major', 'y', ls='-', lw=.5, c='k', alpha=.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just
    # plotted.
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=True, left=False, right=False, labelleft=True)
    # plt.figure(figsize=(10,5))# set fig canvas size
    fig.suptitle(fig_name,fontsize=17,ha="center")
    plt.xlabel(u'rank',fontsize=14)
    plt.ylabel(u'M_A',fontsize=14)

    for i in range(len(spec_row)):
        ax.plot(ma_mat[i,:],linewidth=2,linestyle='-',label=conv_array[i], marker='o')

    plt.legend(bbox_to_anchor=(0.5, -0.06), loc="upper center", ncol=6, frameon=False)
    plt.savefig(save_plot_path + fig_name + ".png", bbox_inches='tight', dpi=100)
    plt.close(fig_name)