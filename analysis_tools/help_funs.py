#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:22:57 2023

@author: sigridtragenap
"""
import numpy as np
import h5py
from matplotlib.colors import hsv_to_rgb
#from skimage.transform import resize


def trial_trial_correlation(trial_data):
    num_trial,num_stim, num_cells = trial_data.shape
    tuning=np.nanmean(trial_data, axis=0)

    res=[]
    indices = np.triu_indices(num_trial, k=1)
    for istim in range(num_stim):
        trial_correlation_stimulus=np.corrcoef(trial_data[:,istim])
        #print(trial_correlation_stimulus.shape)
        assert(trial_correlation_stimulus.shape[0]==num_trial)
        res.append(trial_correlation_stimulus[indices].mean())
    return np.asarray(res)

# def resize_stack(stack, scale=2, **kwargs):
#     if len(stack.shape)>2:
#         result=[]
#         for i in range(stack.shape[0]):
#             result.append(resize_stack(stack[i],scale, **kwargs))
#         return np.asarray(result)

#     w, h = stack.shape
#     shape_new =(w//scale,h//scale)
#     frame_x=resize(stack, output_shape=shape_new,
#                     mode='reflect', order=1)
#     return frame_x

def get_F1_F0(activity):
    num_stim, n_phase = activity.shape[:2]
    fft_coeffs = np.fft.fft(activity, axis=1)
    f1 = np.abs(fft_coeffs[:,1])
    f0 = f0 = np.abs(fft_coeffs[:,0]) #np.mean(activity, axis=1)
    return f0,f1

def opm_from_tuning(tuning):
    num_stim, num_cells = tuning.shape
    angles = np.arange(num_stim)*2*np.pi/float(num_stim)
    angles = np.exp(2j*angles)
    opm = np.nansum(tuning*angles[:,None], axis=0)/(np.nansum(tuning, axis=0)+1e-6)
    return opm

def phi_from_map(opm_est):
    phi_est = 0.5*np.angle(opm_est) + np.pi*(np.angle(opm_est)<0)  #2d
    phi_est[phi_est==np.pi] = 0
    return phi_est

def calc_mismatch(phi, phi_est):
    return np.abs(np.rad2deg(0.5*np.angle(np.exp(2j*(phi - phi_est)))))



def opm_to_rgb(opm, clip):

    def phi_to_hue(phi):
        return phi/np.pi

    def selec_to_S(selec, perc):
        selec=np.clip(selec/perc, a_max=1, a_min=0)
        return selec


    def phi_from_map(opm_est):
        phi_est = 0.5*np.angle(opm_est) + np.pi*(np.angle(opm_est)<0)  #2d
        phi_est[phi_est==np.pi] = 0
        return phi_est

    phi=phi_from_map(opm)
    selec=np.abs(opm)
    H = phi_to_hue(phi)
    S = selec_to_S(selec, clip)
    V = np.ones_like(H)
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def collect_analysis(ana_function, filenames, key_lists):
    res=[]
    if len(filenames)==1:
        for key in key_lists:
            f=h5py.File(filenames[0], "r")
            activity=f[str(key)]["activity"][:,:,-1]
            f.close()
            res.append(ana_function(activity))
    if len(filenames)>1:
        for File in filenames:
            for key in key_lists:
                f=h5py.File(File)
                activity=f[str(key)]["activity"][:,:,-1]
                res.append(ana_function(activity))
                f.close()
    return np.asarray(res)

def collect_analysis_time(ana_function, filenames,
                          key_lists, full_act=False,
                         key_lists_ref=None, **kwargs):
    if key_lists_ref is None:
        key_lists_ref=np.copy(key_lists)
    res=[]
    for File in filenames:
        f=h5py.File(filenames[0], "r")
        for ref, key in zip(key_lists_ref, key_lists):
            if not full_act:
                res_time=[]

                activity=f[str(key)]["activity"][:]
                ref_activity=f[str(ref)]["activity"][:]
                Ntime=activity.shape[2]
                for istep in range(Ntime):
                    res_time.append(ana_function(activity[:,:,istep],
                                                 ref_activity, **kwargs))
                res.append(np.asarray(res_time))
            if full_act:
                activity=f[str(key)]["activity"][:]
                ref_activity=f[str(ref)]["activity"][:]
                res.append(ana_function(activity, ref_activity, **kwargs))
        f.close()
    return np.asarray(res)

def get_all_valid_indices(indices, conditions, file_np):
    list_all_indices=[]
    for index in indices:
        all_clear=True
        for parameter, value in conditions.items():
            all_clear=np.logical_and(all_clear,
                       value==file_np[str(index)][parameter][...])
        if all_clear: list_all_indices.append(index)
    return sorted(list_all_indices)

def get_all_params(index, file_np):
    network_params=file_np[str(index)].keys()
    for parameter in network_params:
        if parameter=="inputs": continue
        if parameter=="eigenvals": continue
        if "ecc" in parameter: continue
        print(parameter, file_np[str(index)][parameter][...])
