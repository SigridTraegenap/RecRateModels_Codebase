#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:10:29 2024

@author: sigridtragenap
"""

def get_tuning_params(conn_mode):
    if conn_mode == 'NaiveAL':
        tuning_params = {
            'orientation_mode': 'modular_random',  # how map should be constructed (correlated, endogen, ...)
            'OrientationMap_sigma': [2, 3],
            'orientation_mode_mod_random': 1,  # how much random if mode is modular_random
            'amplitude_mean': 2.5,
            'amplitude_std': 0,
            'amplitude_DirectionSel': None,
            'amplitude_OriSel': None,
            'amplitude_structure': None,
            'TuningWidth_mean': 58,  # 56 for N 31 for E
            'TuningWidth_std': 8,  # 8 for N 8 for E
            'TuningWidth_Diff': None,
            'baserate_mean': -.5,
            'baserate_std': 0,
            'baserate_max': 3,
            'trialnoise_mean': 1.3,  # 1.8 for N 0.8 for E
            'trialnoise_std': 0,
            'trialnoise_max': 50,
            'trial_noise_mode': 'white_random',  #modular_random
            'noise_mode_mod_random': 1}

    if conn_mode == 'ExperiencedAL':
        tuning_params = {
            'orientation_mode': 'modular_random',  # how map should be constructed (correlated, endogen, ...)
            'OrientationMap_sigma': [2, 3],
            'orientation_mode_mod_random': 1,  # how much random if mode is modular_random
            'amplitude_mean': 2.5,
            'amplitude_std': 0,
            'amplitude_DirectionSel': None,
            'amplitude_OriSel': None,
            'amplitude_structure': None,
            'TuningWidth_mean': 33,  # 56 for N 31 for E
            'TuningWidth_std': 15,  # 8 for N 8 for E
            'TuningWidth_Diff': None,
            'baserate_mean': -1,
            'baserate_std': 0,
            'baserate_max': 3,
            'trialnoise_mean': 2.2,  # 1.8 for N 0.8 for E
            'trialnoise_std': 0,
            'trialnoise_max': 50,
            'trial_noise_mode': 'white_random',
            'noise_mode_mod_random': 1}


    if conn_mode == 'NaiveST':
        tuning_params ={
            'orientation_mode'      : 'modular',  #how map should be constructed (correlated, endogen, ...)
            'OrientationMap_sigma'  : [1,1.6],   #[1,3]   #wavelength of map if mode is modular_random or modular
            #'orientation_mode_mod_random': 0.4,    #how much random if mode is modular_random
            'amplitude_mean'        : 4,
            'amplitude_std'         : 0.5,
            'amplitude_DirectionSel': None,
            'amplitude_structure'   : None,
            'TuningWidth_mean'      : 60,
            'TuningWidth_std'       : 10,
            'TuningWidth_Diff'      : None,
            'baserate_mean'         : 0,
            'baserate_std'          : 0.1,
            'baserate_max'          : 3,
            'trialnoise_mean'       : 1.75,
            'trialnoise_std'        : 0,
            'trialnoise_max'        : 50,
            'trial_noise_mode'      :'white_random',   #bandpass  white_random
            'trial_noise_sigmas'    :[3, 5]
            }




    if conn_mode == 'ExperiencedST':
        tuning_params ={
            'orientation_mode'      : 'modular',  #how map should be constructed (correlated, endogen, ...)
            'OrientationMap_sigma'  : [1,1.6],   #[1,3]   #wavelength of map if mode is modular_random or modular
            #'orientation_mode_mod_random': 0.4,    #how much random if mode is modular_random
            'amplitude_mean'        : 7.5,
            'amplitude_std'         : 0.5,
            'amplitude_DirectionSel': None,
            'amplitude_structure'   : None,
            'TuningWidth_mean'      : 30,
            'TuningWidth_std'       : 25,
            'TuningWidth_Diff'      : None,
            'baserate_mean'         : 1,
            'baserate_std'          : 0.1,
            'baserate_max'          : 3,
            'trialnoise_mean'       : 2,
            'trialnoise_std'        : 1,
            'trialnoise_max'        : 50,
            'trial_noise_mode'      :'white_random',   #bandpass  white_random
            'trial_noise_sigmas'    :[3, 5]
            }

    if conn_mode == 'simple_sinusoid':
        tuning_params ={
            'orientation_mode'      : 'determined_random',  #how map should be constructed (correlated, endogen, ...)
            'OrientationMap_sigma'  : [1,1.6],   #[1,3]   #wavelength of map if mode is modular_random or modular
            'orientation_mode_mod_random': 0.4,    #how much random if mode is modular_random
            'amplitude_mean'        : 1,
            'amplitude_std'         : 0,
            'amplitude_DirectionSel': None,
            'amplitude_structure'   : None,
            'TuningWidth_mean'      : 60,
            'TuningWidth_std'       : 10,
            'TuningWidth_Diff'      : None,
            'baserate_mean'         : 0,
            'baserate_std'          : 0.0,
            'baserate_max'          : 0,
            'trialnoise_mean'       : 0.1,
            'trialnoise_std'        : 0,
            'trialnoise_max'        : 50,
            'trial_noise_mode'      :'white_random',   #bandpass  white_random
            'trial_noise_sigmas'    :[3, 5]
            }

    if conn_mode == 'simple_sinusoid_diverse_ampl':
        tuning_params ={
            'orientation_mode'      : 'determined_random',  #how map should be constructed (correlated, endogen, ...)
            'OrientationMap_sigma'  : [1,1.6],   #[1,3]   #wavelength of map if mode is modular_random or modular
            'orientation_mode_mod_random': 0.4,    #how much random if mode is modular_random
            'amplitude_mean'        : 1,
            'amplitude_std'         : 0.1,
            'amplitude_DirectionSel': None,
            'amplitude_structure'   : None,
            'TuningWidth_mean'      : 60,
            'TuningWidth_std'       : 10,
            'TuningWidth_Diff'      : None,
            'baserate_mean'         : 0,
            'baserate_std'          : 0.0,
            'baserate_max'          : 0,
            'trialnoise_mean'       : 0.1,
            'trialnoise_std'        : 0,
            'trialnoise_max'        : 50,
            'trial_noise_mode'      :'white_random',   #bandpass  white_random
            'trial_noise_sigmas'    :[3, 5]
            }



    return tuning_params

def get_feedforward_connectivity_params(conn_mode, N, M, Seed=1804):
    if conn_mode == '1to1':

        feedforward_connectivity_params={
            "N,M"                   : (N, M),
            'mixing_binary'         : False,  #distance dependence (False) or selected from binary disk
            'radius_FFconnectivity' : 6,   #over how many L4 neurons (radius) input to L2.3 is integrated
            'gauss_sigma'           : 0.1,   #sprad of ff connections
            'sampling'              : False,
            'sampling_K'            : 10,   #typical number of input units to L2.3 unit,
            'eccentricity'          : 0,
            }
    return feedforward_connectivity_params
