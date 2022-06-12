import os
import glob
import random
import scipy.io as sio
import numpy as np
import tensorflow as tf
import math

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # set a GPU (with GPU Number)

PI = math.pi

QF = 'various'
# cover
# path1 = './datasets/alaskav2/jpeg-mat/qf%s'%(QF)
# path2 = './datasets/alaskav2/jpeg-mat/jpeg2spatial/qf%s'%(QF)

# stego
path1 = './datasets/alaskav2/jpeg-mat/qf%s_uerd04'%(QF)
path2 = './datasets/alaskav2/jpeg-mat/jpeg2spatial/qf%s_uerd04'%(QF)

x = tf.placeholder(tf.float32, shape=[1, 256, 256, 1])
tables = tf.placeholder(tf.float32, shape=[1, 256, 256, 1])

# dataQ = sio.loadmat('./quant_tables/quant_' + QF + '.mat')      # the quant table size must be 256*256

xT = tf.multiply(x, tables)
IDCTBase = np.zeros([8, 8, 1, 64], dtype=np.float32)  # [height,width,input,output]
w = np.ones([8], dtype=np.float32)
w[0] = 1.0 / math.sqrt(2.0)
for i in range(0, 8):
    for j in range(0, 8):
        for k in range(0, 8):
            for l in range(0, 8):
                IDCTBase[k, l, :, i * 8 + j] = w[k] * w[l] / 4.0 * math.cos(PI / 16.0 * k * (2 * i + 1)) * math.cos(
                    PI / 16.0 * l * (2 * j + 1))
IDCTKernel = tf.Variable(IDCTBase, name="IDCTKenel", trainable=False)
Pixel = tf.nn.conv2d(xT, IDCTKernel, [1, 8, 8, 1], 'VALID', name="Pixel") + 128
Input = tf.depth_to_space(Pixel, 8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for image in glob.glob(path1 + '/*.mat'):
        data_various = sio.loadmat(image)['im_quant']
        it = data_various[0,0]['im']
        it = np.expand_dims(it, 0)
        it = np.expand_dims(it, 3)

        quant = data_various[0,0]['quant']
        quant = quant.astype(np.float32)
        quant = np.expand_dims(quant, 0)
        quant = np.expand_dims(quant, 3)
        
        fl = sess.run(Input, feed_dict={x: it, tables: quant})
        (_, image_name) = os.path.split(image)
        path = path2 + '/' + image_name
        sio.savemat(path, {'im': fl[0, :, :, 0]})
