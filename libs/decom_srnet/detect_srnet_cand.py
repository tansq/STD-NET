import sys
import os
import tensorflow as tf
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
# from SRNet_2 import *
# from SRNet_7_all import *
from utils import *
import configparser


def get_floats(config_log, layer, index):
    cf = configparser.ConfigParser()
    cf.read(config_log)
    if index == 1:
        ir_ori = cf.getfloat('first_layer_input_ranks', 'layer' + str(layer))
        orr_ori = cf.getfloat('first_layer_output_ranks', 'layer' + str(layer))
    else:
        ir_ori = cf.getfloat('second_layer_input_ranks', 'layer' + str(layer))
        orr_ori = cf.getfloat('second_layer_output_ranks', 'layer' + str(layer))
    return ir_ori, orr_ori


"""
traverse shrinking rate from 1 to 0.2
the matrix rank is reduced by 5% every time
print: the M_A after each reduction
return: M_A_list
"""
def inspect_decom_conv_cand(model_class, gen, batch_size, ds_size, \
                         load_path_ori, layer, index, output_log, F_o_mat, ma_threshold, ranks_result, err, cand, config_log):
                         
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
    err = 0.001
    ir = ranks_result[0]
    orr = ranks_result[1]
    while 1:
        if layer == 2 or layer == 9 and index == 1:
            write_2_config_log(config_log, layer, index, ir, orr)
            break
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
        if cand == 1:
            ir = ir + 1
            orr = orr - 1
        else:
            ir = ir - 1
            orr = orr + 1
        with tf.Session() as sess:
            sess.run(init_op)
            runner.start_threads(sess, 1)
            ranks_1 = []
            ranks_1 = [ir, orr]
            core, [first, last] = partial_tucker(var_ori_1,modes=[2,3],ranks=ranks_1,init='svd')
            first = np.float32(np.expand_dims(np.expand_dims(first.copy(), axis=0),axis=0))
            last = last.transpose((1,0))
            last = np.float32(np.expand_dims(np.expand_dims(last.copy(), axis=0),axis=0))
            assign_sess(sess, load_path_ori, layer, index, first, core, last, var_bias_1)
                
            for j in range(0, ds_size, batch_size):
                sess.run(increment_op)
                # laod the decom_conv+bn from decom_srnet_layerX and get the feature map
                decom_bn = sess.graph.get_tensor_by_name(fc_tensor_name)
                F_c = sess.run(decom_bn)

                F_c = F_c.reshape(batch_size,-1)
                # F_o = np.zeros((1,F_o_mat.shape[1]))
                # F_o = F_o_mat[batch_size,:]

                print("F_c.shape: ", F_c.shape)
                    
                M_A, ma = compute_ma(batch_size, F_o_mat, F_c)
                    


            if ma > threshold + err:
                if cand == 1:
                    write_2_config_log(config_log, layer, index, ir-1, orr+1)
                else:
                    ir_ori, orr_ori = get_floats(config_log, layer, index)
                    if ir_ori * orr_ori > (ir+1) * (orr-1):
                        write_2_config_log(config_log, layer, index, ir+1, orr-1) 
                break
            else:
                output_log.write('input_rank ' + str(ir) + ' output_rank ' + str(orr) +',ma ' + str(ma) + '\n')
                output_log.flush()
                print('ma ', ma)
                print('ranks ', ranks_1)
            if cand == 1:
                if ir == var_ori_1.shape[2] or orr == 1:
                    write_2_config_log(config_log, layer, index, ir, orr)
                    break
            else:
                if ir == 1 or orr == var_ori_1.shape[3]:
                    write_2_config_log(config_log, layer, index, ir, orr)
                    break
                    
                        
            #mat_col = mat_col + 1
            mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                             accuracy_summary.mean_variable])
    
            
