#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:50:42 2017

@author: sigridtragenap

GENERAL INFORMATION ABOUT IMPLEMENTATION:
Any numeric inputs have to be tensorflow variables (that includes activity, proc_inputs, etc.)
"""

import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(0, True)

def forward_euler(inputs,
                  w_rec,
                  start_activity,
                  delta_t,
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float32):
    """ Tensorflow implementation of the forward euler method.
    """

    proc_inputs = inputs[0,:]
    num_neurons = proc_inputs.shape[0]


    # Calculate the activities of all neurons for all times by scanning over "time".
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_frames, num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = np.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity

    def fn_euler(tl_neuronal_activity,proc_input):
        return tl_neuronal_activity + (delta_t / tau) * (
                -tl_neuronal_activity + nonlinearity(tf.tensordot(tl_neuronal_activity, tf.transpose(w_rec), axes=1) + proc_inputs))


    #input
    if inputs.shape[0]==1:
        f_input = tf.tile(inputs, multiples=(k,1))
    else:
        #assert inputs.shape[0]==k
        f_input = tf.identity(inputs)

    res = tf.nest.map_structure(tf.stop_gradient, tf.scan(
            fn=fn_euler,
            elems=f_input,
            initializer=t_start_activity,
            swap_memory=True,
            infer_shape=True,
        ))
    return res



def runge_kutta_explicit(inputs,
                  w_rec,
                  start_activity,
                  delta_t,
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float32):
    """ Tensorflow implementation of the runge kutta method, 4th order.
    For explanation see numpy implementation. """

    proc_inputs = inputs[0,:]
    num_neurons = inputs.shape[0]
    #RK = imrk.Runge_Kutta_Simpson
    #RK = imrk.Runge_Kutta_Fehlberg


    # Calculate the activities of all neurons for all times by scanning over "time".
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_frames, num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = np.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity

    #input
    if inputs.shape[0]==1:
        f_input = tf.tile(inputs, multiples=(k,1))
    else:
        #assert inputs.shape[0]==k
        f_input = tf.identity(inputs)

    def fprime(x, proc_inputs):
        f = (-x + nonlinearity(proc_inputs + tf.tensordot(x, tf.transpose(w_rec), axes=1)))/tau
        return f

    def rk4step(x, proc_inputs):
        k1 = fprime(x, proc_inputs)
        k2 = fprime(x + 0.5*k1*delta_t, proc_inputs)
        k3 = fprime(x + 0.5*k2*delta_t, proc_inputs)
        k4 = fprime(x +     k3*delta_t, proc_inputs)
        x_new = x + delta_t*( (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4 )
        return x_new

    res = tf.nest.map_structure(tf.stop_gradient, tf.scan(
            fn=rk4step,
            elems=f_input,
            initializer=t_start_activity,
            swap_memory=True,
            infer_shape=True,
        ))
    return res




def runge_kutta2(inputs,
                  w_rec,
                  start_activity,
                  delta_t,
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float32):
    """ Tensorflow implementation of the runge kutta method, 2nd order.  """

    proc_inputs = inputs[0,:]
    num_neurons = proc_inputs.shape[0]

    def fprime(x, proc_inputs):
        f = (-x + nonlinearity(proc_inputs + tf.tensordot(x, tf.transpose(w_rec), axes=1)))/tau
        return f

    def rk2step(x, proc_inputs):
        k1 = fprime(x, proc_inputs)
        x_new = x + delta_t*fprime(x+k1*0.5*delta_t, proc_inputs)
        return x_new

    if start_activity is None:
        t_start_activity = np.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity


    #input
    if inputs.shape[0]==1:
        f_input = tf.tile(inputs, multiples=(tf.make_ndarray(k),1))
    else:
        assert inputs.shape[0]==k
        f_input = np.copy(inputs)

    print("start scan")
    res = tf.nest.map_structure(tf.stop_gradient, tf.scan(
            fn=rk2step,
            elems=f_input,
            initializer=t_start_activity,
            swap_memory=True,
            infer_shape=True,
        ))
    return res


