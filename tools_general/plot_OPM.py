#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:37:46 2024

@author: sigridtragenap
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


def opm_from_tuning(tuning, normkey=True):
    num_stim, num_cells = tuning.shape
    if num_stim==9:
        tuning = tuning[:8]
        return opm_from_tuning(tuning)
    if num_stim==17:
        tuning = tuning[:16]
        return opm_from_tuning(tuning)
    angles = np.arange(num_stim)*2*np.pi/float(num_stim)
    angles = np.exp(2j*angles)
    #norm=np.nansum(tuning, axis=0)
    norm=np.nansum(np.abs(tuning), axis=0)
    if normkey==False: norm=1
    opm = np.nansum(tuning*angles[:,None], axis=0)/norm
    #/norm
    return opm

def phi_from_map(opm_est):
    phi_est = 0.5*np.angle(opm_est) + np.pi*(np.angle(opm_est)<0)  #2d
    phi_est[phi_est==np.pi] = 0
    return phi_est

def opm_to_rgb_bright(opm, clip):

    def phi_to_hue(phi):
        return phi/np.pi

    def selec_to_S(selec, th):
        #perc=np.percentile(selec[selec>0], th)
        selec=np.clip(selec/th, a_max=1, a_min=0)
        return selec
        #/np.max(selec)
        #np.clip(selec/(selec_max*th), a_max=1, a_min=0)



    phi=phi_from_map(opm)
    selec=np.abs(opm)
    H = phi_to_hue(phi)
    S = selec_to_S(selec, clip)
    V = np.ones_like(H)
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def opm_to_rgb(opm, clip):

    def phi_to_hue(phi):
        return phi/np.pi

    def selec_to_S(selec, th):
        #perc=np.percentile(selec[selec>0], th)
        selec=np.clip(selec/th, a_max=1, a_min=0)
        return selec
        #/np.max(selec)
        #np.clip(selec/(selec_max*th), a_max=1, a_min=0)


    phi=phi_from_map(opm)
    selec=np.abs(opm)
    H = phi_to_hue(phi)
    V = selec_to_S(selec, clip)
    S = np.ones_like(H)#*0.5
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    return RGB

def opm_to_rgb_muted(opm, clip):

    def phi_to_hue(phi):
        return phi/np.pi

    def selec_to_S(selec, th):
        #perc=np.percentile(selec[selec>0], th)
        selec=np.clip(selec/th, a_max=1, a_min=0)
        return selec
        #/np.max(selec)
        #np.clip(selec/(selec_max*th), a_max=1, a_min=0)


    phi=phi_from_map(opm)
    selec=np.abs(opm)
    H = phi_to_hue(phi)
    V = selec_to_S(selec, clip)
    S = np.ones_like(H)*0.6
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    return RGB


if __name__=='__main__':
    Npix = 100
    opm_plot = np.ones((Npix,Npix), dtype='complex')
    #combine random selectivity and angle
    opm_plot = np.random.rand(Npix,Npix)*np.exp(2j*np.random.rand(Npix,Npix)*2*np.pi)

    opm_frame_in=opm_to_rgb(opm_plot.reshape(Npix,Npix), 1)
    plt.imshow(opm_frame_in)
    plt.title('random  OPM: dark')
    plt.axis('off')
    plt.show()

    opm_frame_in=opm_to_rgb_bright(opm_plot.reshape(Npix,Npix), 1)
    plt.imshow(opm_frame_in)
    plt.title('random  OPM: bright')
    plt.axis('off')
    plt.show()

    opm_frame_in=opm_to_rgb_muted(opm_plot.reshape(Npix,Npix), 1)
    plt.imshow(opm_frame_in)
    plt.title('random  OPM: muted')
    plt.axis('off')
    plt.show()
