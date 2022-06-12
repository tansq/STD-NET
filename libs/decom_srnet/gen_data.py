import numpy as np
from scipy import misc, io
from glob import glob
import random
#from itertools import izip  #there is no itertools in python3.x, zip can be directly used
from random import random as rand
from random import shuffle

import imageio


def gen_flip_and_rot(cover_root, stego_root, cover_txt, stego_txt, thread_idx, n_threads):
    """
    Args:
        cover_root, stego_root: the root paths of cover and stego which are end with '/'
        cover_txt, stego_txt: absolute paths list of sub_dataset
    """
    
    with open(cover_txt) as f:
        cover_list = f.readlines()
        cover_list = [cover_root + a.strip() for a in cover_list]

    with open(stego_txt) as f:
        stego_list = f.readlines()
        stego_list = [stego_root + a.strip() for a in stego_list]

    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_txt
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_txt
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    load_mat=cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['im']
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='float32')
    else:
        img = imageio.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    
    iterable = list(zip(cover_list, stego_list)) 
    while True:
        shuffle(iterable)
        for cover_path, stego_path in iterable:
            if  load_mat:
                batch[0,:,:,0] = io.loadmat(cover_path)['im']
                batch[1,:,:,0] = io.loadmat(stego_path)['im']
            else:
                batch[0,:,:,0] = imageio.imread(cover_path)
                batch[1,:,:,0] = imageio.imread(stego_path)
            rot = random.randint(0,3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]
                              

def gen_valid(cover_root, stego_root, cover_txt, stego_txt, thread_idx, n_threads):
    """
    Args:
        cover_root, stego_root: the root paths of cover and stego which are end with '/'
        cover_txt, stego_txt: absolute paths list of sub_dataset
    """

    with open(cover_txt) as f:
        cover_list = f.readlines()
        cover_list = [cover_root + a.strip() for a in cover_list]

    with open(stego_txt) as f:
        stego_list = f.readlines()
        stego_list = [stego_root + a.strip() for a in stego_list]
        
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_txt
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_txt
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    load_mat=cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['im']
        #img = h5py.File(cover_list[0])['im2']
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='float32')
    else:
        img = imageio.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    img_shape = img.shape
    
    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in list(zip(cover_list, stego_list)):
            if  load_mat:
                batch[0,:,:,0] = io.loadmat(cover_path)['im']
                batch[1,:,:,0] = io.loadmat(stego_path)['im']
            else:
                batch[0,:,:,0] = imageio.imread(cover_path)
                batch[1,:,:,0] = imageio.imread(stego_path)
            yield [batch, labels]
