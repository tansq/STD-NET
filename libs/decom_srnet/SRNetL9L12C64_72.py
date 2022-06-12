import sys
import tensorflow as tf
from functools import partial
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools

from utils import *


class SRNetC64DE72(Model):
    def _build_model(self, inputs):
        self.inputs = inputs
        if self.data_format == 'NCHW':
            reduction_axis = [2, 3]
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1, 2]
            _inputs = tf.cast(inputs, tf.float32)
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None), \
             arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True,
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format), \
             arg_scope([layers.avg_pool2d],
                       kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1'):
                conv = layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
                actv = tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer2'):
                conv = layers.conv2d(actv)
                actv = tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer3'):
                conv1 = layers.conv2d(actv)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(actv, bn2)
            with tf.variable_scope('Layer4'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)
            with tf.variable_scope('Layer5'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer6'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer7'):
                #conv1_1=layers.conv2d(res, num_outputs=16, kernel_size=1, padding='VALID',biases_initializer=None)
                #conv1_2=layers.conv2d(conv1_1, num_outputs=16, biases_initializer=None)
                #conv1_3=layers.conv2d(conv1_2, num_outputs=16, kernel_size=1, padding='VALID')
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2_1=layers.conv2d(actv1, num_outputs=16, kernel_size=1, padding='VALID',biases_initializer=None)
                conv2_2=layers.conv2d(conv2_1, num_outputs=16, biases_initializer=None)
                conv2_3=layers.conv2d(conv2_2, num_outputs=16, kernel_size=1, padding='VALID')
                #conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2_3)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer8'):
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer9'):
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=64)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=64)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer10'):
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=64)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=64)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer11'):
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=64)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=64)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer12'):
                conv1 = layers.conv2d(res, num_outputs=64)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=64)
                bn = layers.batch_norm(conv2)
                avgp = tf.reduce_mean(bn, reduction_axis, keep_dims=True)
        ip = layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                                    activation_fn=None, normalizer_fn=None,
                                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                    biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
        return self.outputs
