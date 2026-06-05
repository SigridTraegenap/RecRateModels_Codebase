#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:50:42 2022

@author: sigridtragenap
"""
import numpy as np


def get_random_field(N,M,N_fields=1, sig1=2, sig2=4, seed=None):
    #constant input in time!
    #N, M : grid size
    #N_fields: number of fields
    #sig1, sig2: kernel size

    #generate random input
    rng_fields = np.random.default_rng(seed)
    #sample from normal distribution, other distributions possible
    input_rnd = rng_fields.normal(size=[N_fields, N, M])

    if np.allclose(0,sig1):
        bp_field = np.copy(input_rnd)
    else:
        #apply convolution
        #use convolution with MH to get spatial scale in noisy input
        #define convolutio kernels
        x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N),np.linspace(-M//2+1,M//2,M))
        kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
        kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
        input_smo = np.real(np.fft.ifft2(np.fft.fft2(
            kern1-kern2)[None,:,:]*np.fft.fft2(
                input_rnd,axes=(1,2)), axes=(1,2)))
        bp_field = input_smo.copy()

    bp_field -= np.nanmean(bp_field, axis=(-1,-2))[:,None,None]
    bp_field /= np.nanstd(bp_field, axis=(-1,-2))[:,None,None]
    return bp_field


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    A=get_random_field(100,100,N_fields=10, sig1=2, sig2=4, seed=None)
    print(A.shape)

    plt.imshow(A[0], cmap="PRGn")
    plt.colorbar()
    plt.show()
