#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:11:11 2024

@author: sigridtragenap


Model for a "brain network". Template for use with numpy (np) etc.
used to be in python3
"""

from os.path import abspath, sep, pardir
import sys
sys.path.append(abspath('') + sep + pardir + sep + pardir + sep)
import numpy
#import numpy.distutils
#import numpy.distutils.__config__
###
import numpy as np
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

from tqdm import tqdm
import time


import types
print("numpy -V", numpy.__version__)
print("tf -V", tf.__version__)

from . import bn_tools_tf as bnt
from . import integration_methods_tf as im


rng_noise = np.random.RandomState(1)


# Network support
class BrainNetwork:
    def __init__(self,
                w_rec,
                nonlinearity_rule,
                integrator='forward_euler',
                delta_t=0.01,
                tau=10.,
                tsteps=None,
                data_type=np.float32):

        if type(data_type) == str:
            data_type = np.typeDict[data_type]
        self.data_type_np = data_type
        num_neurons = w_rec.shape[0]
        self.num_neurons = num_neurons

        # Convert variables to new data type

        if w_rec is None:
            w_rec = np.zeros((num_neurons, num_neurons))
        w_rec = self.data_type_np(w_rec)


        delta_t = self.data_type_np(delta_t)
        tau = self.data_type_np(tau)
        if tsteps is not None:
            tsteps = self.data_type_np(tsteps)

        self._tf_init = False
        self.w_rec = w_rec.astype(self.data_type_np)
        self.delta_t = np.float32(delta_t)

        if tsteps is not None:
            self.tsteps = np.int32(tsteps)
        else:
            self.tsteps = None

        if isinstance(tau, np.ndarray):
            self.tau = tau.astype(self.data_type_np)
        else:
            self.tau = np.float32(tau)

        #kw to choose functions later
        self.nonlinearity_rule = nonlinearity_rule
        self.integrator = integrator

        #tf constants
        self.tf_w_rec = tf.constant(self.w_rec, name="w_rec")
        self.tf_delta_t = tf.constant(self.delta_t, name="deltat")
        self.tf_tsteps = tf.constant(self.tsteps, name="tsteps")
        self.tf_tau = tf.constant(self.tau, name="tau")

        #tf variables
        self.tf_inputs =tf.constant(np.zeros(shape=(1,num_neurons)), name="inputs")#, dtype=tf.float32)
        self.tf_start_activity = tf.constant(np.zeros(shape=(num_neurons)), name="startA")#, dtype=tf.float32)

        #graph compilation
        self._init_nonlinearity()
        self._init_integrator()
        self._init_tf_computations()


    def update_w_rec(self, w_rec):
        self.t_w_rec.set_value(w_rec.astype(self.data_type_np))
        #if (not self.t_w_rec is None) and (w_rec.shape == self.t_w_rec.get_value().shape):
        #    print 'WARNING (bn_t): w_rec potentially as wrong shape'

    def _tf_function_update(self):
        self._tf_init = False

    def _init_nonlinearity(self):
        if self.nonlinearity_rule == 'linear':
            self.tf_nonlinearity = bnt.nl_linear_tf
        elif self.nonlinearity_rule == 'rectification':
            self.tf_nonlinearity = bnt.nl_rect_tf
        elif self.nonlinearity_rule == 'rectification_threshold':
            self.tf_nonlinearity = bnt.nl_rect_th_tf
        elif self.nonlinearity_rule == 'shallow_rectification':
            self.tf_nonlinearity = bnt.nl_shallow_rect_tf
        elif self.nonlinearity_rule == 'clipping':
            self.tf_nonlinearity = bnt.nl_clip_tf
        elif self.nonlinearity_rule == 'sigmoid':
            self.tf_nonlinearity = bnt.nl_sigmoid_tf
        elif self.nonlinearity_rule == 'sigmoid_v2':
            self.tf_nonlinearity = bnt.nl_sigmoidcustom_tf
        elif self.nonlinearity_rule == 'tanh':
            self.tf_nonlinearity = bnt.nl_tanh_tf
        elif self.nonlinearity_rule =='fermi':
            self.tf_nonlinearity = bnt.nl_fermi_tf
        elif self.nonlinearity_rule =='rectification_squared':
            self.tf_nonlinearity = bnt.nl_rect_squared_tf

        else:
            raise Exception('Unknown nonlinearity rule')
        self._tf_function_update()   #set _tf_init to False #set function in first iteration


    def inspect_inputs(self,i, node, fn):
        print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs], end='')

    def inspect_outputs(self,i, node, fn):
        print(" output(s) value(s):", [output[0] for output in fn.outputs])



    def _init_integrator(self):
        if self.integrator == 'runge_kutta':
            self.integrator_function = im.runge_kutta_explicit
        elif self.integrator == 'runge_kutta2':
            self.integrator_function = im.runge_kutta2
        elif self.integrator == 'forward_euler':
            self.integrator_function = im.forward_euler
        else:
            raise Exception("Unknown integrator ({0})".format(self.integrator))
        self._tf_function_update()

    def _init_tf_computations(self):

        if not self._tf_init:
            # Compute the activity by plugging the variables into an external
            # integrator function.

            @tf.function
            def activity(tf_inputs, tf_start_activity):
                return self.integrator_function(tf_inputs,
                                self.tf_w_rec,
                                tf_start_activity,
                                self.tf_delta_t,
                                self.tf_tau,
                                self.tf_nonlinearity,
                                data_type_np=self.data_type_np,
                                k=self.tf_tsteps)

            self.tf_activity = activity

            self._tf_init = True

    def run(self, inputs, start_activity):

        if isinstance(inputs, types.FunctionType) or isinstance(inputs, types.MethodType):
            # Inputs is a function
            print("unknown input as Function or Method")
        else:
            # Inputs is an array
            inputs = tf.cast(inputs, dtype=(self.data_type_np), name="inputs")

        start_activity = tf.cast(start_activity, dtype=(self.data_type_np), name="startAI") #typecasting
        self._check_variables()
        activity = self.tf_activity(inputs,start_activity)
        return activity

    def _check_variables(self):
        if (self.tf_w_rec is None or
            self.nonlinearity_rule is None or
            self.integrator is None or
            self.tf_delta_t is None or
            self.tf_tau is None):
            raise Exception("Not all required settings set.")

    def res_Input_mat(self, inputs,
                      total_t,
                      timepoints=2,
                      start=None):
        nsim_total, nneurons = inputs.shape

        if start is None:
            start_activity = np.zeros(nneurons)


        activity = np.empty((nsim_total, timepoints, nneurons))*np.nan

        ## Run network to get spont. activity pattern
        #rng_sa = np.random.RandomState(index*2)
        print( "run!");sys.stdout.flush()
        for i in tqdm(range(nsim_total)):


             ## simulate
             #start2 = time.time()
             iinput = inputs[i].reshape(1,nneurons)
             iactivity = self.run(iinput, start_activity)

             #end2 = time.time()
             #print("Time elapsed", end2 - start2)
             #print(iactivity.shape, total_t, N, M)
             iactivity = (iactivity.numpy()).reshape(total_t,nneurons)
             ## shape is runtime x N x M

             ## store only specific time points of simulated activity traces
             activity[i,:,:] = iactivity[total_t//timepoints-1::total_t//timepoints,:]

        return activity


class PlasticBrainNetwork(BrainNetwork):
    def __init__(self,
                w_rec,
                nonlinearity_rule,
                integrator='forward_euler',
                delta_t=0.01,
                tau=10.,
                tsteps=None,
                data_type=np.float32,
                tau_weights=10000,  #learning rate
                tau_threshold=50000, #timescale threshold
                act_prev_layer=None, #activity of L4
                w_feedforward=None, #Feedforward weights,
                desired_firing_rate=1,
                num_neurons_PreviousLayer=4,
                threshold_init=1,
                ):
        super().__init__(w_rec, nonlinearity_rule, integrator, delta_t, tau,
                tsteps, data_type)  #initialize parent class
        #initialize new elements
        self.num_neurons_PreviousLayer=num_neurons_PreviousLayer
        self.tau_threshold=1.*tau_threshold
        self.tau_weights= 1.*tau_weights
        self.desired_firing_rate= desired_firing_rate


        if act_prev_layer is None:
            act_prev_layer=np.ones(num_neurons_PreviousLayer,
                          dtype=self.data_type_np)
        self.act_prev_layer = act_prev_layer.astype(self.data_type_np)

        if w_feedforward is None:
            w_feedforward=np.ones([self.num_neurons,self.num_neurons_PreviousLayer],
                          dtype=self.data_type_np)
        self.w_feedforward = w_feedforward.astype(self.data_type_np)

        #initial threshold (can be average activity over one spont event) initialize externally
        self.plastic_thresholds = threshold_init*np.ones(self.num_neurons,
                                                   dtype=self.data_type_np)

    #compare ELife paper Wosniak, ..,Gjorgjeva
    def _fprime_weights(self,weights_project,kwargs):
        activity_current=kwargs["activity_current"]
        timescale_learn= (1./self.tau_weights)
        activity_postsyn = activity_current*(activity_current-self.plastic_thresholds)
        weight_change=np.dot(activity_postsyn[:,None], self.act_prev_layer[None,:])
        return timescale_learn*weight_change

    def _fprime_thresholds(self,thresholds, kwargs):
        activity_current=kwargs["activity_current"]
        timescale_th= (1./self.tau_threshold)
        update= np.square(activity_current)/self.desired_firing_rate - thresholds
        return timescale_th*update

    def _euler_forward(self, x, fprime, delta_t, **kwargs ):
        x_new= x + delta_t*fprime(x, kwargs)
        return x_new

    def _runge_kutta_explicit(self,x, fprime,delta_t, **kwargs):
        k1 = fprime(x, kwargs)
        k2 = fprime(x + 0.5*k1*delta_t, kwargs)
        k3 = fprime(x + 0.5*k2*delta_t, kwargs)
        k4 = fprime(x + k3*delta_t, kwargs)
        x_new = x + delta_t*( (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4 )
        return x_new

    def update_feedforwardweights(self, activity_current):
        delta_t = np.asarray(self.delta_t * self.tf_tsteps.numpy(), dtype=self.data_type_np)



        #calc BCM update for weights
        old_w=np.copy(self.w_feedforward)
        self.w_feedforward = self._euler_forward(old_w,
                                     self._fprime_weights, delta_t,
                                     activity_current=activity_current)
        self.w_feedforward[self.w_feedforward<0]=0.


        #calc threshold
        old_theta=np.copy(self.plastic_thresholds)
        self.plastic_thresholds= self._euler_forward(old_theta,
                                    self._fprime_thresholds, delta_t,
                                    activity_current=activity_current)

    def get_input_toL2(self, act_prev_layer):
        self.act_prev_layer=act_prev_layer
        return np.dot(self.w_feedforward, self.act_prev_layer )

