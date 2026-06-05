#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:11:11 2024

@author: sigridtragenap
"""

import numpy as np


# Nonlinearity functions (Numpy implementation)
nl_linear = lambda x: x
nl_tanh = lambda x: np.tanh(x)
nl_sigmoid = lambda x: 1./(1+np.exp(-x))
nl_rect = lambda x: np.clip(x, 0, np.inf)
#nl_rect = lambda x: np.clip(x, -np.inf, np.inf)
nl_shallow_rect = lambda x: np.clip(0.1*x, 0, np.inf)
nl_clip = lambda x: np.clip(x, 0, 1)
nl_softplus = lambda x: np.log(1. + np.exp(x)) #

# Nonlinearity functions (TensorFlow implementation)

#import numpy.distutils
import tensorflow as tf

nl_linear_tf = lambda x: x
nl_tanh_tf = lambda x: tf.tanh(x)
nl_sigmoid_tf = lambda x: tf.sigmoid(x)
nl_fermi_tf = lambda x: tf.sigmoid(x*50)
nl_clip_tf = lambda x: tf.clip_by_value(x, 0., 1.)
nl_rect_tf = lambda x: tf.nn.relu(x)
nl_rect_th_tf = lambda x: tf.nn.relu(x-0.1)
nl_rect_squared_tf = lambda x: tf.nn.relu(x**2)
nl_shallow_rect_tf = lambda x:  tf.nn.relu(0.1*x)
#nl_sigmoidcustom_tf = lambda x: 5/(1+tf.math.exp(-0.8*x+5))
nl_sigmoidcustom_tf = lambda x: tf.nn.relu(tf.sigmoid(x) - 0.5)*2
#lambda x: 2/(1+tf.math.exp(-tf.nn.relu(x))) - 1
#tf.nn.relu(tf.sigmoid(x) - 0.5)*2

def setup_sigmoid(max_S, slope, offset):
	return lambda x: max_S/(1+np.exp(-4*slope*(x-offset)))

def convert_input_const_to_time(inp, num_frames):
    if inp.shape[0] != 1:
        raise Exception("First axis of inp has to be 1-dim.")
    if inp.shape[1] != 1:
        inp = inp[:, 0:1, :]
        print('WARNING (bn_tools): Input has more than one frame. Only first frame will be broadcast.')

    inp = np.tile(inp, (1, num_frames, 1))
    return inp

def check_nonlinearities():
    import matplotlib.pyplot as plt
    x_np=np.arange(-5,5,0.1).astype('float32')
    x=np.copy(x_np)
    for fkt in [nl_clip_tf,nl_sigmoid_tf]:
        plt.plot(x_np, fkt(x))
    plt.show()

if __name__=='__main__':
    check_nonlinearities()
