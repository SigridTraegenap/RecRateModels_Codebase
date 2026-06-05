#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 00:10:05 2023

@author: sigridtragenap
"""

import scipy.ndimage as snd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#https://stackoverflow.com/questions/59685140/
#python-perform-blur-only-within-a-mask-of-image

def idealFilterBP(low_bp,high_bp,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    xv, yv = np.meshgrid(np.arange(rows), np.arange(cols))
    positions=np.dstack([xv,yv]).reshape(rows*cols,2)
    center_point = np.asarray([[rows/2,cols/2]])

    distance_matrix = distance.cdist(positions,center_point)
    base = (distance_matrix<high_bp)*(distance_matrix>low_bp)
    #center = (rows/2,cols/2)

    # for x in range(cols):
    #     for y in range(rows):
    #         if (distance((y,x),center) < high_bp) and (distance((y,x),center) > low_bp):
    #             base[y,x] = 1
    return base.reshape(rows, cols)

def get_additivenoise_ideal(full_shape,index=None,
                                   npatterns=1,seed=81622,
                                   dist_btw_peaks_min=None,
                                   dist_btw_peaks_max=None):
    if dist_btw_peaks_max is None:
        dist_btw_peaks_max=dist_btw_peaks_min+5
    input_shape_woneuron = full_shape[:-2]
    N,M = full_shape[-2], full_shape[-1]
    nevents = np.prod(input_shape_woneuron)


    spat_freq,spat_freq_upper = N/dist_btw_peaks_max, N/dist_btw_peaks_min
    ideal_filt = idealFilterBP(spat_freq,spat_freq_upper,(N,M))

    rng = np.random.default_rng(seed)
    #np.random.seed(seed)
    patterns=[]
    #print(full_shape)
    for ipattern in range(nevents):
        img_c1 = rng.normal(0, 1, size=(N,M))
        img_c2 = np.fft.fft2(img_c1)
        img_c3 = np.fft.fftshift(img_c2)

        img_c3=img_c3*ideal_filt


        img_c4 = np.fft.ifftshift(img_c3)
        img_c5 = np.fft.ifft2(img_c4)
        patterns.append(img_c5.real)
    patterns = np.asarray(patterns)
    patterns = patterns.reshape(input_shape_woneuron+(N,M,))

    return np.asarray(patterns)


def lowhigh_filter_stack(frame, mask=None,sig_high=2, sig_low=15):
    boundary_mode='wrap'
    if len(frame.shape)>2:
        result=np.empty_like(frame)
        for i in range(frame.shape[0]):
            result[i]=lowhigh_filter_stack(frame[i],mask,
                                        sig_high, sig_low)
        return result

    if np.iscomplexobj(frame):
        result=np.empty_like(frame)
        result.real=lowhigh_filter_stack(frame.real,mask,sig_high,sig_low)
        result.imag=lowhigh_filter_stack(frame.imag,mask,sig_high,sig_low)
        return result

    if mask is None:
        mask=np.ones(frame.shape,dtype=float)

    mask[~np.isfinite(frame)]=0.
    frame[np.logical_not(mask)] = 0.

    filter_img_low = snd.gaussian_filter(frame * mask,
                               sigma = sig_low,mode=boundary_mode)
    weights_low = snd.gaussian_filter(mask,
                  sigma = sig_low,mode=boundary_mode)
    filter_img_low /= (weights_low)
    #filter_img_low /= (weights_low+1e-10)


    filter_img_high = snd.gaussian_filter(frame * mask,
                      sigma = sig_high,mode=boundary_mode)
    weights_high = snd.gaussian_filter(mask,
                      sigma = sig_high,mode=boundary_mode)
    filter_img_high /= (weights_high+1e-10)
    # filter_img_high /= weights_high


    highlow_data = filter_img_low-filter_img_high
    highlow_data[np.logical_not(mask)]=np.nan
    return highlow_data


if __name__=='__main__':
    import Modular_2D.analysis.analysis_tools.help_funs as hf
    N=60

    rng = np.random.default_rng()  #184

    pref_ori = rng.uniform(low=0, high=360, size=(N,N) )

    opm_prefori = np.exp(2j*np.deg2rad(pref_ori))

    opm_frame = hf.opm_to_rgb(opm_prefori, 1)

    plt.subplot(321)
    plt.imshow(opm_frame)
    plt.axis('off')

    plt.subplot(322)
    plt.hist(np.rad2deg(
        hf.phi_from_map(opm_prefori.flatten()))
        )
    plt.axis('off')

    sigma=2.8
    opm_filtered = lowhigh_filter_stack(opm_prefori,
                                        sig_high=2.6, sig_low=sigma)
    opm_frame = hf.opm_to_rgb(opm_filtered, 0)

    plt.subplot(323)
    plt.imshow(opm_frame)
    plt.axis('off')

    plt.subplot(324)
    plt.hist(np.rad2deg(
        hf.phi_from_map(opm_prefori.flatten()))
        )
    plt.axis('off')

    plt.subplot(325)
    opm_rolled = np.roll(opm_frame,(N//2,N//2), axis=(0,1,))

    plt.imshow(opm_rolled)
    plt.axis('off')



    plt.show()

