#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:13:50 2023

@author: sigridtragenap
"""
import numpy as np


def SinuisoidGrating(Size, A=1, angle=0,
                     frequency_Kspace=10,
                     phase=0, Pos=[0,0],):
    # Generate Sinusoid grating
    #Size
    x = np.arange(-Size/2.0,Size/2.,1)
    X, Y = np.meshgrid(x, x)

    grating = A*np.cos(
       frequency_Kspace*X*np.cos(angle) +
       frequency_Kspace*Y*np.sin(angle) -
       phase)
    return grating

def getGabor(Size, angle=0, frequency_Kspace=10, sigmas=[10,10], phase=0,
             Pos=(0,0),):

    x = np.arange(-Size/2.0,Size/2.,1)
    X, Y = np.meshgrid(x, x)

    gauss = 1./(2*np.pi*sigmas[0]*sigmas[1])*np.exp(- X**2/(2*sigmas[0]**2)
                                                    -Y**2/(2*sigmas[1]**2 ))
    cosine = np.cos(
        frequency_Kspace*X*np.cos(angle) +
        frequency_Kspace*Y*np.sin(angle) -
        phase)

    gabor = gauss*cosine
    return gabor/np.abs(gabor).sum()

def getEllipsoid(Size, Pos=(0,0), angle=0, sigmas=(10,10)):
    x = np.arange(-Size/2.0,Size/2.,1)
    X, Y = np.meshgrid(x, x)

    a = np.cos(angle)**2/(2*sigmas[0])+np.sin(angle)**2/(2*sigmas[1])
    b = -np.sin(2*angle)/(4*sigmas[0])+np.sin(2*angle)/(4*sigmas[1])
    c = np.sin(angle)**2/(2*sigmas[0])+np.cos(angle)**2/(2*sigmas[1])

    gauss = np.exp(- (a*(X)**2 +
                      c*(Y)**2 +
                    2*b*(X)*(Y)
                     )
                   )
    return gauss/np.abs(gauss).sum()



def get_responses(receptive_field,
                  stims=np.arange(0,180,22.5),
                  Kstim=1/6):

    Size=receptive_field.shape[0]
    res=[]
    for iangle in stims:
        stim = SinuisoidGrating(Size, A=1, angle=np.deg2rad(iangle),
                             frequency_Kspace=Kstim,
                             phase=0,)
        res.append([np.sum((stim*receptive_field))])
    return np.asarray(res)
