import sys
import os
import tensorflow as tf
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
# from SRNet_2 import *
# from SRNet_7_all import *
from utils import *



def inspect_decom_conv_before_cand(model_class, gen, batch_size, ds_size, \
                         load_path_ori, layer, index, output_log, F_o_mat, ma_threshold, config_log):
    
    print("Ready to traverse shrinking rate in l%d-conv%d" % (layer, index))
    output_log.write('--------layer' + str(layer) + '--------' + time.strftime("%Y%m%d-%H%M%S") + '\n')
    output_log.write('------' + str(index) + '------' + '\n') 
    
    var_name, var_name_bias, fc_tensor_name = get_decom_tensor_name(layer, index)
    var_ori_1 = tf.contrib.framework.load_variable(load_path_ori, var_name)
    var_bias_1 = tf.contrib.framework.load_variable(load_path_ori, var_name_bias)
    if layer == 2:
        threshold = ma_threshold[0]
    else:
        threshold = ma_threshold[(layer-3)*2+index]
    

    if layer == 2 or (layer == 9 and index == 1):
        #step = 0.05
        for i in range(0,16):   # rate reduction from 1 to 0.20 
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
                if layer == 2:
                    ranks = [var_ori_1.shape[2] - 4 * i, var_ori_1.shape[3] - i]
                else:
                    ranks = [var_ori_1.shape[2] - i, var_ori_1.shape[3] - 4 * i] 
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
                    M_A, ma = compute_ma(batch_size, F_o_mat[j:j+batch_size,:], F_c)  
                    
                print("Rank: {}\n".format(ranks))
                if ma > threshold:
                    write_2_config_log(config_log, layer, index, ranks_result[0], ranks_result[1])
                    return ranks_result
                    break
                else:
                    ranks_result = ranks
                    output_log.write("Rank: {}, ma :{}\n".format(ranks, ma))
                    output_log.flush()
                    sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])

    elif (layer >= 3 and layer <= 8) or (layer == 9 and index == 2) or (layer >= 10 and layer <= 12):
    
        ma_dim = 64
        
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
                    M_A, ma = compute_ma(batch_size, F_o_mat[j:j+batch_size,:], F_c)
                    
                
                mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable,
                                                     accuracy_summary.mean_variable])
                print("Rank: {}".format(ranks))
                if ma > threshold:
                    write_2_config_log(config_log, layer, index, ranks_result[0], ranks_result[1])
                    return ranks_result
                    break
                else:
                    ranks_result = ranks
                    output_log.write("Rank: {}, ma={} \n".format(ranks, ma))
                    output_log.flush()
                    sess.run([loss_summary.reset_variable_op, accuracy_summary.reset_variable_op])
    
        