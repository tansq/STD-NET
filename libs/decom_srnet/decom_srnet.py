import sys
import os
import tensorflow as tf
import tensorly
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import configparser

def decom_ori_srnet_one_layer_first(load_path, layer, r1, r2): #decompose the first convolution layer of the blocks in srnet such as Layer3/Conv/weights
    if layer >= 3 and layer <= 7 or layer == 12 or layer == 2:
        var_name = 'Layer' + str(layer) + '/Conv/weights'
        var_name_bias = 'Layer' + str(layer) + '/Conv/biases'
        var = tf.contrib.framework.load_variable(load_path, var_name)
        var_bias = tf.contrib.framework.load_variable(load_path, var_name_bias)
        ranks = [r1, r2]        
        core, [first, last] = partial_tucker(var,modes=[2,3],ranks=ranks,init='svd')
        first=np.float32(np.expand_dims(np.expand_dims(first.copy(), axis=0),axis=0))
        last=last.transpose((1,0))
        last=np.float32(np.expand_dims(np.expand_dims(last.copy(), axis=0),axis=0))
    elif layer >=8 and layer<=11:
        var_name = 'Layer' + str(layer) + '/Conv_1/weights'
        var_name_bias = 'Layer' + str(layer) + '/Conv_1/biases'
        var = tf.contrib.framework.load_variable(load_path, var_name)
        var_bias = tf.contrib.framework.load_variable(load_path, var_name_bias)
        ranks = [r1, r2]        
        core, [first, last] = partial_tucker(var,modes=[2,3],ranks=ranks,init='svd')
        first=np.float32(np.expand_dims(np.expand_dims(first.copy(), axis=0),axis=0))
        last=last.transpose((1,0))
        last=np.float32(np.expand_dims(np.expand_dims(last.copy(), axis=0),axis=0))
    return first, core, last, var_bias

def decom_ori_srnet_one_layer_second(load_path, layer, r1, r2): #decomposed the second convolution layer of the blocks in srner such as :Layer3/Conv_1/weights
    if layer >= 3 and layer <= 7 or layer == 12:
        var_name = 'Layer' + str(layer) + '/Conv_1/weights'
        var_name_bias = 'Layer' + str(layer) + '/Conv_1/biases'
        var = tf.contrib.framework.load_variable(load_path, var_name)
        var_bias = tf.contrib.framework.load_variable(load_path, var_name_bias)
        ranks = [r1, r2]        
        core, [first, last] = partial_tucker(var,modes=[2,3],ranks=ranks,init='svd')
        first=np.float32(np.expand_dims(np.expand_dims(first.copy(), axis=0),axis=0))
        last=last.transpose((1,0))
        last=np.float32(np.expand_dims(np.expand_dims(last.copy(), axis=0),axis=0))
    elif layer >=8 and layer<=11:
        var_name = 'Layer' + str(layer) + '/Conv_2/weights'
        var_name_bias = 'Layer' + str(layer) + '/Conv_2/biases'
        var = tf.contrib.framework.load_variable(load_path, var_name)
        var_bias = tf.contrib.framework.load_variable(load_path, var_name_bias)
        ranks = [r1, r2]        
        core, [first, last] = partial_tucker(var,modes=[2,3],ranks=ranks,init='svd')
        first=np.float32(np.expand_dims(np.expand_dims(first.copy(), axis=0),axis=0))
        last=last.transpose((1,0))
        last=np.float32(np.expand_dims(np.expand_dims(last.copy(), axis=0),axis=0))
    return first, core, last, var_bias

def copy_weights_to_decom_srnet(load_path_ori, load_path_new, config_path, log_path, model_name): #copy the decomposed weights to new model
    first_1 = []
    core_1 = []
    last_1 = []
    bias_1 = []
    first_2 = []
    core_2 = []
    last_2 = []
    bias_2 = []
    for layer in range(2,13):
        if layer == 2:
            cf = configparser.ConfigParser()
            cf.read(config_path)
            r1 = cf.getfloat('first_layer_input_ranks', 'layer' + str(layer))
            r2 = cf.getfloat('first_layer_output_ranks', 'layer' + str(layer))
            (first_var_1, core_var_1, last_var_1, var_bias_1) = decom_ori_srnet_one_layer_first(load_path_ori, layer, int(r1), int(r2))
            first_1.append(first_var_1)
            core_1.append(core_var_1)
            last_1.append(last_var_1)
            bias_1.append(var_bias_1)
        else:   
            cf = configparser.ConfigParser()
            cf.read(config_path)
            r1 = cf.getfloat('first_layer_input_ranks', 'layer' + str(layer))
            r2 = cf.getfloat('first_layer_output_ranks', 'layer' + str(layer))
            (first_var_1, core_var_1, last_var_1, var_bias_1) = decom_ori_srnet_one_layer_first(load_path_ori, layer, int(r1), int(r2))
            r1 = cf.getfloat('second_layer_input_ranks', 'layer' + str(layer))
            r2 = cf.getfloat('second_layer_output_ranks', 'layer' + str(layer))
            (first_var_2, core_var_2, last_var_2, var_bias_2) = decom_ori_srnet_one_layer_second(load_path_ori, layer, int(r1), int(r2))
            first_1.append(first_var_1)
            core_1.append(core_var_1)
            last_1.append(last_var_1)
            bias_1.append(var_bias_1)
            first_2.append(first_var_2)
            core_2.append(core_var_2)
            last_2.append(last_var_2)
            bias_2.append(var_bias_2)
    tf.reset_default_graph()
    new_var_list=[]
    new_name_list=[]
    name_list=[]
    for var_name, _ in tf.contrib.framework.list_variables(load_path_ori):
        name_list.append(var_name)
    with tf.Session() as sess:
        for layer in range(2,13):
            if layer == 2:
                new_name = 'Layer' + str(layer) + '/Conv/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(first_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_1/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(core_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_2/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(last_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                var = tf.contrib.framework.load_variable(load_path_ori, 'Layer' + str(layer) + '/Conv/biases')
                new_name_list.append('Layer' + str(layer) + '/Conv_2/biases')
                new_var_list.append(tf.Variable(var,name='Layer' + str(layer) + '/Conv_2/biases'))
                continue

            elif layer >=3 and layer<=7 or layer == 12:
                new_name = 'Layer' + str(layer) + '/Conv/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(first_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_1/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(core_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_2/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(last_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                var = tf.contrib.framework.load_variable(load_path_ori, 'Layer' + str(layer) + '/Conv/biases')
                new_name_list.append('Layer' + str(layer) + '/Conv_2/biases')
                new_var_list.append(tf.Variable(var,name='Layer' + str(layer) + '/Conv_2/biases'))
                new_name = 'Layer' + str(layer) + '/Conv_3/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(first_2[layer-3], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_4/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(core_2[layer-3], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_5/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(last_2[layer-3], name=new_name)
                new_var_list.append(renamed_var)
                var = tf.contrib.framework.load_variable(load_path_ori, 'Layer' + str(layer) + '/Conv_1/biases')
                new_name_list.append('Layer' + str(layer) + '/Conv_5/biases')
                new_var_list.append(tf.Variable(var,name='Layer' + str(layer) + '/Conv_5/biases'))
                continue
 
            elif layer >= 8 and layer <= 11:
                new_name = 'Layer' + str(layer) + '/Conv_1/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(first_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_2/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(core_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_3/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(last_1[layer-2], name=new_name)
                new_var_list.append(renamed_var)
                var = tf.contrib.framework.load_variable(load_path_ori, 'Layer' + str(layer) + '/Conv_1/biases')
                new_name_list.append('Layer' + str(layer) + '/Conv_3/biases')
                new_var_list.append(tf.Variable(var,name='Layer' + str(layer) + '/Conv_3/biases'))
                new_name = 'Layer' + str(layer) + '/Conv_4/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(first_2[layer-3], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_5/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(core_2[layer-3], name=new_name)
                new_var_list.append(renamed_var)
                new_name = 'Layer' + str(layer) + '/Conv_6/weights'
                new_name_list.append(new_name)
                renamed_var=tf.Variable(last_2[layer-3], name=new_name)
                new_var_list.append(renamed_var)
                var = tf.contrib.framework.load_variable(load_path_ori, 'Layer' + str(layer) + '/Conv_2/biases')
                new_name_list.append('Layer' + str(layer) + '/Conv_6/biases')
                new_var_list.append(tf.Variable(var,name='Layer' + str(layer) + '/Conv_6/biases'))
                continue
                 
                
        for var_name, _ in tf.contrib.framework.list_variables(load_path_new):
            if var_name in new_name_list:
                continue 
            elif var_name in name_list and len(var_name.split('/')) <= 3:
                var = tf.contrib.framework.load_variable(load_path_ori, var_name) 
                new_var_list.append(tf.Variable(var, name=var_name))
            else:
                var = tf.contrib.framework.load_variable(load_path_new, var_name) 
                new_var_list.append(tf.Variable(var, name=var_name))
               
        print('starting to write new checkpoint !') 
        saver = tf.train.Saver(var_list=new_var_list) 
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(log_path):
            os.makedirs(log_path) 
        checkpoint_path = os.path.join(log_path, model_name) 
        saver.save(sess, checkpoint_path)
        
    
    print('done!')

            
