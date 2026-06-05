#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:11:11 2024

@author: sigridtragenap
"""

import numpy as np
from scipy import linalg

class BrainNetworkLinear:
    def __init__(self,
                w_rec,
                ):


        num_neurons = w_rec.shape[0]
        self.num_neurons = num_neurons
        # self.tau_r = tau_neur
        # self.timescales=1-self.eigvals

        self.W_connect = w_rec
        self.Interaction_mat_tp =(linalg.inv(np.eye(self.num_neurons)
                                             -self.W_connect)).T



    def transform_sigma_input(self, sigma_in):
        return self.Interaction_mat @ sigma_in @ self.Interaction_mat.T


    def res_Input_mat(self, input_M):
        assert input_M.shape[-1]==self.num_neurons
        #assert len(input_M.shape)==2
        return input_M @ self.Interaction_mat_tp
