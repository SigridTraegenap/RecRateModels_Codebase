#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:17:44 2023

@author: sigridtragenap
"""
import numpy as np
from . import generate_noisy_mh

def get_paramdict_connectivity(conn_mode,
                               **kwargs):

    if conn_mode=='NN_18_elongMH':
        eccentricity = 0.8  ## eccentricity
        gamma = 1.02 ## gamma/recurrent strength
        print('Parameter settings are:\n ecc={}, gamma={}'.format(eccentricity,gamma))
        network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'postsyn',
            'sigmax'                  : 1.8,
            'sigmax_sd'               : eccentricity*0.15/0.8,
            'ecc'                     : eccentricity,
            'ecc_sd'                  : 0.13*eccentricity,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'amplitude'               : 1.,
            'inh_factor'              : 2.5,
            'sigma_x_input'           : 1.8,
            'pbc'                     :True,
            'generating_function'     :generate_noisy_mh.noisy_mh_wrap
            }

    if conn_mode=='NN_18_elongMH_smooth':
        eccentricity = 0.2  ## eccentricity
        gamma = 1.02 ## gamma/recurrent strength
        print('Parameter settings are:\n ecc={}, gamma={}'.format(eccentricity,gamma))
        network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'postsyn',
            'sigmax'                  : 1.8,
            'sigmax_sd'               : eccentricity*0.15/0.8,
            'ecc'                     : eccentricity,
            'ecc_sd'                  : 0.13*eccentricity,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'amplitude'               : 1.,
            'inh_factor'              : 2.5,
            'sigma_x_input'           : 1.8,
            'pbc'                     :True,
            'generating_function'     :generate_noisy_mh.noisysmooth_mh_wrap
            }

    if conn_mode=='homogenMH':
        eccentricity = 0.0  ## eccentricity
        gamma = 1.02 ## gamma/recurrent strength
        print('Parameter settings are:\n ecc={}, gamma={}'.format(eccentricity,gamma))
        network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'None',
            'sigmax'                  : 1.8,
            'sigmax_sd'               : 0,
            'ecc'                     : 0,
            'ecc_sd'                  : 0,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'amplitude'               : 1.,
            'inh_factor'              : 2.5,
            'sigma_x_input'           : 1.8,
            'pbc'                     :True,
            'generating_function'     :generate_noisy_mh.noisy_mh_wrap

            }
    if conn_mode=='NN_18_elongMH_smaller':
            eccentricity = 0.8  ## eccentricity
            gamma = 1.02 ## gamma/recurrent strength
            #print('Parameter settings are:\n ecc={}'.format(eccentricity))
            network_params = {
                'mode'                    : 'short_range',
                'noise_type'              : 'postsyn',
                'sigmax'                  : 3.6,
                'sigmax_sd'               : eccentricity*0.15/0.8,
                'ecc'                     : eccentricity,
                'ecc_sd'                  : 0.13*eccentricity,
                'orientation'             : 0,
                'orientation_sd'          : 1.0,
                'amplitude'               : 1.,
                'inh_factor'              : 2.5,
                'sigma_x_input'           : 1.8,
                'pbc'                     :True,
                }

    if conn_mode=='NN_18_MH':
            eccentricity = 0.0  ## eccentricity
            gamma = 1.02 ## gamma/recurrent strength
            #print('Parameter settings are:\n ecc={}, gamma={}'.format(eccentricity))
            network_params = {
                'mode'                    : 'short_range',
                'noise_type'              : 'postsyn',
                'sigmax'                  : 1.8,
                'sigmax_sd'               : eccentricity*0.15/0.8,
                'ecc'                     : eccentricity,
                'ecc_sd'                  : 0.13*eccentricity,
                'orientation'             : 0,
                'orientation_sd'          : 1.0,
                'amplitude'               : 1.,
                'inh_factor'              : 2.5,
                'sigma_x_input'           : 1.8,
                'pbc'                     :True,
                }

    if conn_mode=='EI_MH':
        eccentricity = 0.  ## eccentricity
        network_params = {
            'mode' : 'short_range',
            'noise_type' : 'postsyn',
            'sigmax': 2,
            'sigmax_sd': eccentricity * 0.15 / 0.8,
            'ecc' : eccentricity,
            'ecc_sd' : 0.13*eccentricity,
            'orientation' : 0,
            'ori_sd' : 1,
            'sum_ei' : 1,
            'sum_ii' : 1.,
            'inh_factor' : 2.5,
            'pbc'                     :True,
           'generating_function'     :generate_noisy_mh.noisy_EI_wrap
            }

    if conn_mode=='MH_PlusRF':
           network_params = {
               'mode'                    : 'short_range',
               'noise_type'              : 'None',
               'sigmax'                  : 1.8,
               #'sigmax_sd'               : eccentricity*0.15/0.8,
               #'ecc'                     : eccentricity,
               #'ecc_sd'                  : 0.13*eccentricity,
               'orientation'             : 0,
               'orientation_sd'          : 1.0,
               'amplitude'               : 1.,
               'inh_factor'              : 2.5,
               'sigma_x_input'           : 1.8,
               'pbc'                     :True,
               'sigma1_bp'               :2,
               'sigma2_bp'               :6,
               'strength_RF'             :0.3,
               'perturb_samepat'         :False,
               'generating_function'     :generate_noisy_mh.homogenMH_plusRF_wrap
               }



    if conn_mode == 'Mod_mult_dym':
        network_params = {
            'Dist_Spec': 5, #7.5
            'Mod_High': 2*2.5,
            'Mod_Low': 2,
            'gamma': 2.5, # 3
            'dym': 0, # 3,
            'generating_function'     :generate_noisy_mh.multi_dym_modular_wrap
            }

    if conn_mode == 'modular_conn_vS':
        network_params = {
            'Dist_Spec': 5, #7.5
            'Mod_High': 2*2.5,
            'Mod_Low': 2,
            'gamma': 2.5, # 3
            'beta_M': 0, # 3,
            'zscore': True,
            'generating_function'     :generate_noisy_mh.modular_comp_wrap
            }


    if conn_mode=='NComm24_MH_PlusRF':
        network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'None',
            'sigmax'                  : 1.8,
#             'sigmax_sd'               : eccentricity*0.15/0.8,
#             'ecc'                     : eccentricity,
#             'ecc_sd'                  : 0.13*eccentricity,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'amplitude'               : 1.,
            'inh_factor'              : 2,
            'sigma_x_input'           : 1.8,
            'pbc'                     :True,
            'sigma1_bp'               :2,
            'sigma2_bp'               :6,
            'strength_RF'             :0.4,
            'perturb_samepat'         :True,
            'generating_function'     :generate_noisy_mh.homogenMH_plusRF_wrap,
            'normalization'           :False,
            'normalization_fallback'  :True,
            'return_pertubfield'      :False,
            }

    if conn_mode=='EI_deformedMH':
        network_params = {
            'mode' : 'short_range',
            'noise_type' : 'None',
            'sigmax': 2,
            'sum_ei' : 1,
            'sum_ii' : 1.,
            'inh_factor' : 2.5,
            'pbc'                     :True,
            'sigma1_bp'               :2,
            'sigma2_bp'               :6,
            'strength_RF'             :0.4,
            'perturb_samepat'         :True,
           'generating_function'     :generate_noisy_mh.deformed_EI_wrap
            }

    if conn_mode=='EI_MH_selfinhib_pertub':
        eccentricity = 0.  ## eccentricity
        network_params = {
            'mode' : 'short_range',
            'noise_type' : 'postsyn',
            'sigmax': 2,
            'sigmax_sd': eccentricity * 0.15 / 0.8,
            'ecc' : eccentricity,
            'ecc_sd' : 0.13*eccentricity,
            'orientation' : 0,
            'ori_sd' : 1,
            'sum_ei' : 1,
            'sum_ii' : 1,
            'sum_ie' : 1,
            'sum_ee' : 1.,
            'inh_factor' : 2.5,
            'pbc'                     :True,
            'self_inhibition': True,
            'alpha': 0.5,
            'a_tot': 1,
            'sigma1_bp'               :2,
            'sigma2_bp'               :6,
            'strength_RF'             :0.4,
           'generating_function'     :generate_noisy_mh.noisy_EI_wrap
            }

    if conn_mode == 'Stage1_ExcOnly':
        # amplitude=0 → no inhibitory term at all; inh_factor is irrelevant but
        # set finite (not np.inf) to avoid the uniform-mh2 code path in noisy_mh.
        network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'None',
            'sigmax'                  : 1,
            'amplitude'               : 0,
            'inh_factor'              : 1.5,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'sigma_x_input'           : 1.8,
            'pbc'                     : True,
            'sigma1_bp'               : 1,
            'sigma2_bp'               : 3,
            'strength_RF'             : 0.4,
            'perturb_samepat'         : True,
            'normalization'           : False,
            'normalization_fallback'  : True,
            'return_pertubfield'      : False,
            'generating_function'     : generate_noisy_mh.homogenMH_plusRF_wrap,
            }

    return network_params
