
import sys
import configparser
import tensorflow as tf
from functools import partial
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools

from utils import *



def fill_rank(config_log):
    cf = configparser.ConfigParser()
    cf.read(config_log)
    first_layer_input_ranks = []
    first_layer_output_ranks = []
    second_layer_input_ranks = []
    second_layer_output_ranks = []
    for i in range(2, 13):
        if i == 2:
            first_layer_input_ranks.append(cf.getfloat('first_layer_input_ranks', 'layer' + str(i)))
            first_layer_output_ranks.append(cf.getfloat('first_layer_output_ranks', 'layer' + str(i)))
        else:    
            first_layer_input_ranks.append(cf.getfloat('first_layer_input_ranks', 'layer' + str(i)))
            first_layer_output_ranks.append(cf.getfloat('first_layer_output_ranks', 'layer' + str(i)))
            second_layer_input_ranks.append(cf.getfloat('second_layer_input_ranks', 'layer' + str(i)))
            second_layer_output_ranks.append(cf.getfloat('second_layer_output_ranks', 'layer' + str(i)))
    return first_layer_input_ranks, first_layer_output_ranks, second_layer_input_ranks, \
        second_layer_output_ranks

class SRNetDecom(Model):
    def _build_model(self, inputs, config_log):
        self.inputs = inputs
        self.config_log = config_log
        (first_layer_input_ranks, first_layer_output_ranks, second_layer_input_ranks, \
            second_layer_output_ranks) = fill_rank(self.config_log)
        if self.data_format == 'NCHW':
            reduction_axis = [2,3]
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1,2]
            _inputs = tf.cast(inputs, tf.float32)
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None),\
            arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1'): 
                conv=layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
                actv=tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer2'):
                #conv=layers.conv2d(actv) 
                conv_1=layers.conv2d(actv, num_outputs=first_layer_input_ranks[0], kernel_size=1, biases_initializer=None)
                conv_2=layers.conv2d(conv_1, num_outputs=first_layer_output_ranks[0], biases_initializer=None)
                conv_3=layers.conv2d(conv_2, num_outputs=16, kernel_size=1, padding='valid')
                actv=tf.nn.relu(layers.batch_norm(conv_3))
            with tf.variable_scope('Layer3'): 
                #conv1=layers.conv2d(actv)
                conv1_1=layers.conv2d(actv, num_outputs=first_layer_input_ranks[1], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[1], biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2=layers.conv2d(actv1)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[0], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[0], biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=16,kernel_size=1, padding='valid')
                bn2=layers.batch_norm(conv2_3)
                res= tf.add(actv, bn2)
            with tf.variable_scope('Layer4'): 
                #conv1=layers.conv2d(res)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[2], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[2], biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2=layers.conv2d(actv1)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[1], kernel_size=1, padding='valid', biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[1], biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=16,kernel_size=1, padding='valid')
                bn2=layers.batch_norm(conv2_3)
                res= tf.add(res, bn2)
            with tf.variable_scope('Layer5'): 
                #conv1=layers.conv2d(res)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[3], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[3],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2=layers.conv2d(actv1)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[2], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[2],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=16,kernel_size=1, padding='valid')
                bn=layers.batch_norm(conv2_3)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer6'): 
                #conv1=layers.conv2d(res)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[4], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[4],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2=layers.conv2d(actv1)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[3], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[3],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=16,kernel_size=1, padding='valid')
                bn=layers.batch_norm(conv2_3)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer7'): 
                #conv1=layers.conv2d(res)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[5], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[5],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2=layers.conv2d(actv1)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[4], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[4],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=16,kernel_size=1, padding='valid')
                bn=layers.batch_norm(conv2_3)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer8'): 
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                #conv1 = layers.conv2d(res, num_outputs=16)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[6], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[6],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2 = layers.conv2d(actv1, num_outputs=16)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[5], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[5],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=16,kernel_size=1, padding='valid')
                bn=layers.batch_norm(conv2_3)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer9'):  
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                #conv1 = layers.conv2d(res, num_outputs=64)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[7], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[7],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=64, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2 = layers.conv2d(actv1, num_outputs=64)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[6], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[6],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=64,kernel_size=1, padding='valid')
                bn=layers.batch_norm(conv2_3)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer10'): 
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                #conv1 = layers.conv2d(res, num_outputs=128)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[8], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[8],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=64, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                #conv2 = layers.conv2d(actv1, num_outputs=128)
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[7], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[7],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2,num_outputs=64,kernel_size=1, padding='valid')
                bn=layers.batch_norm(conv2_3)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer11'): 
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[9], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[9],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=64, kernel_size=1, padding='valid')
                #conv1 = layers.conv2d(res, num_outputs=256)
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[8], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[8],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2, num_outputs=64, kernel_size=1, padding='valid')
                #conv2 = layers.conv2d(actv1, num_outputs=256)
                bn=layers.batch_norm(conv2_3)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer12'):
                #conv1 = layers.conv2d(res, num_outputs=512) 
                conv1_1=layers.conv2d(res, num_outputs=first_layer_input_ranks[10], kernel_size=1, padding='valid',biases_initializer=None)
                conv1_2=layers.conv2d(conv1_1, num_outputs=first_layer_output_ranks[10],biases_initializer=None)
                conv1_3=layers.conv2d(conv1_2, num_outputs=64, kernel_size=1, padding='valid')
                actv1=tf.nn.relu(layers.batch_norm(conv1_3))
                conv2_1=layers.conv2d(actv1, num_outputs=second_layer_input_ranks[9], kernel_size=1, padding='valid',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=second_layer_output_ranks[9],biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2, num_outputs=64, kernel_size=1, padding='valid')
                #conv2 = layers.conv2d(actv1, num_outputs=512)
                bn=layers.batch_norm(conv2_3)
                avgp = tf.reduce_mean(bn, reduction_axis,  keep_dims=True )
        ip=layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                    activation_fn=None, normalizer_fn=None,
                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                    biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
        return self.outputs
