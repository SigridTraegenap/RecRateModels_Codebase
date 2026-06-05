#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:11:11 2024

@author: sigridtragenap
"""



from os.path import abspath, sep, pardir
import sys
path_parent = abspath('') + sep + pardir + sep + pardir + sep
sys.path.append(path_parent)

import numpy as np
import scipy.linalg as la
from scipy.spatial import distance
import h5py
from .global_params import global_path
import tools_general.help_funs as hf

def shuffle_frame(act_frame,  nmaps=None):
    f_fft=np.fft.rfft2
    f_ifft=np.fft.irfft2

    rng = np.random.RandomState()

    fft=f_fft(act_frame, axes=(0,1))
    absfft=np.abs(fft)
    count=1 if nmaps is None else nmaps
    result=np.empty((count,)+act_frame.shape,dtype=act_frame.dtype)*np.nan
    for i in range(count):
        angles=np.angle(fft)
        rng.shuffle(angles.flat)
        fft_shuffled=absfft*np.exp(1j*angles)
        shuffled=f_ifft(fft_shuffled, s=act_frame.shape, axes=(0,1))
        result[i]=shuffled
    return result[0] if nmaps is None else result


def get_additive_ideal(full_shape,  index,
                       spat_freq=None):
    pass


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

def euclid_distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

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


def get_additive_epsilon_sigma(full_shape, network_params, index, noise_level=None):
    #constant input in time!
    input_shape_woneuron = full_shape[:-2]
    N,M = full_shape[-2], full_shape[-1]
    nevents = np.prod(input_shape_woneuron)


    sig1 = 0.9*noise_level
    sig2 = 1.1*sig1


    #generate random input
    rng_start_activity = np.random.default_rng(index)
    input_rnd = rng_start_activity.random(size=[nevents, N, M]) #UNiform

    if np.allclose(0,sig1):
        additive_noise = np.copy(input_rnd)
    else:
        #apply convolution
        #use convolution with MH to get spatial scale in noisy input
        #define convolutio kernels
        x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N),np.linspace(-M//2+1,M//2,M))
        kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
        kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
        input_smo = np.real(np.fft.ifft2(np.fft.fft2(kern1-kern2)[None,:,:]*np.fft.fft2(input_rnd,axes=(1,2)), axes=(1,2)))
        additive_noise = input_smo
    #reshape to fit original specifified shape
    #print(input_shape_woneuron,N,M, nevents, additive_noise.shape)
    additive_noise = additive_noise.reshape(nevents,N*M)
    additive_noise = (additive_noise - (additive_noise.mean(-1)[:,None]))/additive_noise.std(-1)[:,None]
    additive_noise = additive_noise.reshape(input_shape_woneuron+(N*M,))
    return additive_noise


def get_additive_noise(full_shape, network_params, index={},
                       noise_level=None, seed=0):
    #constant input in time!
    input_shape_woneuron = full_shape[:-2]
    N,M = full_shape[-2], full_shape[-1]
    nevents = np.prod(input_shape_woneuron)

    if noise_level is None:
        ## Input parameters
        sig1 = network_params['sigma_x_input']
        sig2 = 2*sig1
    else:
        sig1 = noise_level
        sig2 = 2*sig1


    #generate random input
    rng_start_activity = np.random.default_rng(index)
    input_rnd = rng_start_activity.random(size=[nevents, N, M]) #UNiform

    if np.allclose(0,sig1):
        additive_noise = np.copy(input_rnd)
    else:
        #apply convolution
        #use convolution with MH to get spatial scale in noisy input
        #define convolutio kernels
        x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N),np.linspace(-M//2+1,M//2,M))
        kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
        kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
        input_smo = np.real(np.fft.ifft2(np.fft.fft2(kern1-kern2)[None,:,:]*np.fft.fft2(input_rnd,axes=(1,2)), axes=(1,2)))
        additive_noise = input_smo
    #reshape to fit original specifified shape
    #print(input_shape_woneuron,N,M, nevents, additive_noise.shape)
    additive_noise = additive_noise.reshape(nevents,N*M)
    additive_noise = (additive_noise - (additive_noise.mean(-1)[:,None]))/additive_noise.std(-1)[:,None]
    additive_noise = additive_noise.reshape(input_shape_woneuron+(N*M,))
    return additive_noise

def load_activity(filename, folder_index_old):
    f = h5py.File(global_path+"{}.hdf5".format(filename), 'r')
    #take last timeframe for all spontaneous patterns
    act_frame = f[str(folder_index_old)]['activity'][:]
    f.close()
    return act_frame


def get_endogenous_nonbinar_pattern(n_stimuli,
                           endogen_mode, network_params,
                           filename,
                           folder_index_old="0/",
                           sigma=4,
                           w_rec=None):

    #n_stimuli: how many patterns to output
    N_old, M_old=network_params['input_shape']
    act_frame = load_activity(filename, folder_index_old)
    #print(act_frame.shape)
    n_trial, n_stim, n_tp, n_pop, N,M = act_frame.shape
    act_frame = act_frame[:,:,-1,0].reshape(n_trial*n_stim,N,M)
    #print('2',act_frame.shape)
    #flat activity

    if endogen_mode=="w_recmodes":
        idx = np.asarray(np.linspace(0,100, num=n_stimuli), dtype='int32')

        U,S,V=la.svd(w_rec)
        new_maps = V[idx].reshape(n_stimuli,N,M)

    if endogen_mode=="purePC":
        idx = np.asarray(np.linspace(0,100, num=n_stimuli), dtype='int32')


        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.cov(act_frames.T)
        U,S,V=la.svd(cov)

        new_maps = V[idx].reshape(n_stimuli,N,M)


    if endogen_mode=="purePC_firstN":
        idx = np.asarray(np.arange(n_stimuli), dtype='int32')


        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.cov(act_frames.T)
        U,S,V=la.svd(cov)

        new_maps = V[idx].reshape(n_stimuli,N,M)


    if endogen_mode=="purePC_shifted":
        idx = np.asarray(np.linspace(0,100, num=n_stimuli), dtype='int32')


        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.cov(act_frames.T)
        U,S,V=la.svd(cov)

        raw_maps = V[idx].reshape(n_stimuli,N,M)

        new_maps=[]
        for i in range(n_stimuli):
            shift_x = np.random.choice(np.arange(1,7))
            shift_y = np.random.choice(np.arange(1,7))

            new_map_x = np.roll(raw_maps[i],shift_x,axis=1)
            new_map_y = np.roll(new_map_x,shift_y,axis=1)
            new_maps.append(new_map_y)

        new_maps = np.asarray(new_maps).reshape(n_stimuli,N,M)

    if endogen_mode=="purePC_phase":
       idx = np.asarray(np.linspace(0,100, num=n_stimuli), dtype='int32')


       act_frames=act_frame.reshape(act_frame.shape[0],-1)
       cov = np.cov(act_frames.T)
       U,S,V=la.svd(cov)

       raw_maps = V[idx].reshape(n_stimuli,N,M)

       new_maps=[]
       for i in range(n_stimuli):

            new_map_y = shuffle_frame(raw_maps[i],
                                      nmaps=1)
            new_maps.append(new_map_y)
       new_maps = np.asarray(new_maps).reshape(n_stimuli,N,M)




    if "random"==endogen_mode:
        #get pattern with similar (heterogenous) modular structure
        #but full spectral
        act_frame_single = act_frame[np.random.choice(act_frame.shape[0])]
        new_maps = shuffle_frame(act_frame_single, nmaps=n_stimuli)


    if "BP"==endogen_mode:
        # new_maps = get_additive_epsilon_sigma(
        #     (n_stimuli,N,M), {}, None,
        #     noise_level=sigma)
        # new_maps=new_maps.reshape(n_stimuli,N,M)
        # print(new_maps.shape)

        #print(N, M)
        new_maps = get_additivenoise_ideal(full_shape=(n_stimuli,N,M),
                                      index=None,
                                           npatterns=n_stimuli,
                                           seed=81622,
                                           spat_freq=sigma,
                                           spat_freq_upper=sigma+1)
        #print(new_maps.shape)
    if "PrincComp"==endogen_mode:

        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])
        base_maps=eigenVectors[:,:2].T
        base_maps=base_maps.reshape(2,N,M)


        alphas=np.arange(n_stimuli)/n_stimuli*np.pi
        new_maps = np.sin(2*alphas)[:,None,None]*base_maps[0][None,:,:] + np.cos(
            2*alphas)[:,None,None]*base_maps[1][None,:,:]

    if "PCs_space"==endogen_mode:
        dim_inputs=20
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])

        # C_input = hf.create_n_dim_gauss(dim_inputs,
        #                                 base_patterns=eigenVectors,
        #                                 n_neurons=N*M)

        # new_maps = hf.sample_from_normal(C_input,
        #              n_samples=n_stimuli)



        Npasebats=200

        eigvals_broad = np.exp(-np.arange(Npasebats)/(0.5*dim_inputs))
        eigvals_broad *= Npasebats/eigvals_broad.sum()

        component_factors = np.random.normal(
                loc=np.zeros(Npasebats),
                scale=eigvals_broad,
                size=(n_stimuli, Npasebats,))

        new_maps = eigenVectors[:,:Npasebats]@component_factors.T
        new_maps=(new_maps.T).reshape(n_stimuli,N,M)




    if "CorrPattern"==endogen_mode:
        #print(act_frame.shape)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        correlation_pat = np.corrcoef(act_frames.T)
        #print(act_frames.shape)
        #print(correlation_pat.shape)

        #choose random seeds for corr patterns (not in box)
        inner_border=10
        outer_border=N-10
        random_seeds_xy = np.random.choice(np.arange(inner_border,outer_border),
                                          size=(2,n_stimuli))
        seeds_flat = np.ravel_multi_index(random_seeds_xy, (N,M))

        #print(np.max(seeds_flat))
        #seelct correlation patterns as basis
        new_maps = correlation_pat[seeds_flat,:]
        new_maps = np.reshape(new_maps, (n_stimuli,N,M))

    if "Activity"==endogen_mode:
        new_maps = act_frame[np.random.choice(act_frame.shape[0], size=n_stimuli)]


    if "shifted"==endogen_mode:
        new_maps_raw = act_frame[np.random.choice(act_frame.shape[0], size=n_stimuli)]

        new_maps=[]
        for i in range(n_stimuli):
            shift_x = np.random.choice(np.arange(1,7))
            shift_y = np.random.choice(np.arange(1,7))

            new_map_x = np.roll(new_maps_raw[i],shift_x,axis=1)
            new_map_y = np.roll(new_map_x,shift_y,axis=1)
            new_maps.append(new_map_y)

    if "CorrPattern_shifted"==endogen_mode:
        #print(act_frame.shape)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        correlation_pat = np.corrcoef(act_frames.T)
        #print(act_frames.shape)
        #print(correlation_pat.shape)

        #choose random seeds for corr patterns (not in box)
        inner_border=10
        outer_border=N-10
        random_seeds_xy = np.random.choice(np.arange(inner_border,outer_border),
                                          size=(2,n_stimuli))
        seeds_flat = np.ravel_multi_index(random_seeds_xy, (N,M))

        #print(np.max(seeds_flat))
        #seelct correlation patterns as basis
        new_maps_raw = correlation_pat[seeds_flat,:]
        new_maps_raw = np.reshape(new_maps_raw, (n_stimuli,N,M))

        new_maps=[]
        for i in range(n_stimuli):
            shift_x = np.random.choice(np.arange(1,7))
            shift_y = np.random.choice(np.arange(1,7))

            new_map_x = np.roll(new_maps_raw[i],shift_x,axis=1)
            new_map_y = np.roll(new_map_x,shift_y,axis=1)
            new_maps.append(new_map_y)


    #zscore patterns
    new_maps /= new_maps.std(axis=(1,2))[:,None,None]
    return new_maps.reshape(n_stimuli, N*M)

def get_endogenous_pattern(n_stimuli,
                           endogen_mode, network_params,
                           filename,
                           folder_index_old="0/",
                           spont_space_btw_peaks_max=15,
                           spont_space_btw_peaks_min=11,
                           w_rec=None):

    #n_stimuli: how many patterns to output


    N_old, M_old=network_params['input_shape']
    act_frame = load_activity(filename, folder_index_old)
    #print(act_frame.shape)
    n_trial, n_stim, n_tp, n_pop, N,M = act_frame.shape
    act_frame = act_frame[:,:,-1,0].reshape(n_trial*n_stim,N,M)
    #print('2',act_frame.shape)
    #flat activity

    if endogen_mode=="w_recmodes":
        idx = np.asarray(np.linspace(0,100, num=n_stimuli), dtype='int32')

        U,S,V=la.svd(w_rec)
        new_maps = V[idx].reshape(n_stimuli,N,M)

    if endogen_mode=="purePC":
        idx = np.asarray(np.linspace(0,100, num=n_stimuli), dtype='int32')


        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.cov(act_frames.T)
        U,S,V=la.svd(cov)

        new_maps = V[idx].reshape(n_stimuli,N,M)

    if "PCs_space"==endogen_mode:
        dim_inputs=20
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])

        # C_input = hf.create_n_dim_gauss(dim_inputs,
        #                                 base_patterns=eigenVectors,
        #                                 n_neurons=N*M)

        # new_maps = hf.sample_from_normal(C_input,
        #              n_samples=n_stimuli)



        Npasebats=200

        eigvals_broad = np.exp(-np.arange(Npasebats)/(0.5*dim_inputs))
        eigvals_broad *= Npasebats/eigvals_broad.sum()

        component_factors = np.random.normal(
                loc=np.zeros(Npasebats),
                scale=eigvals_broad,
                size=(n_stimuli, Npasebats,))

        new_maps = eigenVectors[:,:Npasebats]@component_factors.T
        new_maps=(new_maps.T).reshape(n_stimuli,N,M)

    if endogen_mode=="purePC_firstN":
        idx = np.asarray(np.arange(n_stimuli), dtype='int32')


        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.cov(act_frames.T)
        U,S,V=la.svd(cov)

        new_maps = V[idx].reshape(n_stimuli,N,M)


    if "random" in endogen_mode:
        #get pattern with similar (heterogenous) modular structure
        #but full spectral
        act_frame_single = act_frame[np.random.choice(act_frame.shape[0])]
        new_maps = shuffle_frame(act_frame_single, nmaps=n_stimuli)


    if "BP" in endogen_mode:
        # new_maps = get_additive_epsilon_sigma(
        #     (n_stimuli,N,M), {}, None,
        #     noise_level=sigma)
        # new_maps=new_maps.reshape(n_stimuli,N,M)
        # print(new_maps.shape)

        #print(N, M)
        new_maps = get_additivenoise_ideal(full_shape=(n_stimuli,N,M),
                                      index=None,
                                           npatterns=n_stimuli,
                                           seed=81622,
            dist_btw_peaks_max=spont_space_btw_peaks_max,
            dist_btw_peaks_min=spont_space_btw_peaks_min)
        #print(new_maps.shape)
    if "PrincComp" in endogen_mode:

        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])
        base_maps=eigenVectors[:,:2].T
        base_maps=base_maps.reshape(2,N,M)


        alphas=np.arange(n_stimuli)/n_stimuli*np.pi
        new_maps = np.sin(2*alphas)[:,None,None]*base_maps[0][None,:,:] + np.cos(
            2*alphas)[:,None,None]*base_maps[1][None,:,:]

    if "PCs_space" in endogen_mode:
        dim_inputs=20
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])

        # C_input = hf.create_n_dim_gauss(dim_inputs,
        #                                 base_patterns=eigenVectors,
        #                                 n_neurons=N*M)

        # new_maps = hf.sample_from_normal(C_input,
        #              n_samples=n_stimuli)

        Npasebats=200
        component_factors = np.random.normal(
                loc=np.zeros(Npasebats),
                scale=eigenValues[:Npasebats].real,
                size=(n_stimuli, Npasebats,))

        new_maps = eigenVectors[:,:Npasebats]@component_factors.T
        new_maps=(new_maps.T).reshape(n_stimuli,N,M)




    if "CorrPattern" in endogen_mode:
        #print(act_frame.shape)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        correlation_pat = np.corrcoef(act_frames.T)
        #print(act_frames.shape)
        #print(correlation_pat.shape)

        #choose random seeds for corr patterns (not in box)
        inner_border=10
        outer_border=N-10
        random_seeds_xy = np.random.choice(np.arange(inner_border,outer_border),
                                          size=(2,n_stimuli))
        seeds_flat = np.ravel_multi_index(random_seeds_xy, (N,M))

        #print(np.max(seeds_flat))
        #seelct correlation patterns as basis
        new_maps = correlation_pat[seeds_flat,:]
        new_maps = np.reshape(new_maps, (n_stimuli,N,M))

    if "Activity" in endogen_mode:
        new_maps = act_frame[np.random.choice(act_frame.shape[0], size=n_stimuli)]


    if "shifted" in endogen_mode:
        new_maps_raw = act_frame[np.random.choice(act_frame.shape[0], size=n_stimuli)]

        new_maps=[]
        for i in range(n_stimuli):
            shift_x = np.random.choice(np.arange(1,7))
            shift_y = np.random.choice(np.arange(1,7))

            new_map_x = np.roll(new_maps_raw[i],shift_x,axis=1)
            new_map_y = np.roll(new_map_x,shift_y,axis=1)
            new_maps.append(new_map_y)

    if "CorrPattern_shifted" in endogen_mode:
        #print(act_frame.shape)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        correlation_pat = np.corrcoef(act_frames.T)
        #print(act_frames.shape)
        #print(correlation_pat.shape)

        #choose random seeds for corr patterns (not in box)
        inner_border=10
        outer_border=N-10
        random_seeds_xy = np.random.choice(np.arange(inner_border,outer_border),
                                          size=(2,n_stimuli))
        seeds_flat = np.ravel_multi_index(random_seeds_xy, (N,M))

        #print(np.max(seeds_flat))
        #seelct correlation patterns as basis
        new_maps_raw = correlation_pat[seeds_flat,:]
        new_maps_raw = np.reshape(new_maps_raw, (n_stimuli,N,M))

        new_maps=[]
        for i in range(n_stimuli):
            shift_x = np.random.choice(np.arange(1,7))
            shift_y = np.random.choice(np.arange(1,7))

            new_map_x = np.roll(new_maps_raw[i],shift_x,axis=1)
            new_map_y = np.roll(new_map_x,shift_y,axis=1)
            new_maps.append(new_map_y)


    #zscore patterns
    new_maps -= np.percentile(new_maps, 68, axis=(1,2))[:,None,None]
    #new_maps -= new_maps.std(axis=(1,2))[:,None,None]

    endogen_stimuli = 1.*(new_maps>0)  #-0.5
    return endogen_stimuli.reshape(n_stimuli, N*M)


def get_SCM(kw,network_params,index,filename, folder_index_old="0/"):

    N, M=network_params['input_shape']

    if "random" in kw:
        act_frame = load_activity(filename, folder_index_old)
        act_frame = act_frame[np.random.choice(act_frame.shape[0])]
        new_maps = shuffle_frame(act_frame, index, nmaps=2)
    if "rand_SP" in kw:
        #N,M = act_frame.shape[-2:]
        rng= np.random.default_rng()
        new_maps = rng.normal(size=[2, N, M])
    if "PrincComp" in kw:
        act_frame = load_activity(filename, folder_index_old)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])
        new_maps=eigenVectors[:,:2].T
        new_maps=new_maps.reshape(2,N,M)
    if "LowVarPC_old" in kw:
        act_frame = load_activity(filename, folder_index_old)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])
        new_maps=eigenVectors[:,20:22].T
        new_maps=new_maps.reshape(2,N,M)
    if "LowVarPC_34" in kw:
        act_frame = load_activity(filename, folder_index_old)
        act_frames=act_frame.reshape(act_frame.shape[0],-1)
        cov = np.corrcoef(act_frames.T)
        eigenValues, eigenVectors = la.eig(cov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = np.real(eigenVectors[:,idx])
        new_maps=eigenVectors[:,3:5].T
        new_maps=new_maps.reshape(2,N,M)
    #zero-mean and zero-std
    new_maps = new_maps.reshape(2,-1)
    new_maps = (new_maps - (new_maps.mean(-1)[:,None]))/new_maps.std(-1)[:,None]
    nstim=network_params['nstim']
    alphas=np.arange(nstim)/nstim*np.pi

    stim_input = np.sin(2*alphas)[:,None]*new_maps[0][None,:] + np.cos(2*alphas)[:,None]*new_maps[1][None,:]
    stim_input = stim_input.reshape(nstim, -1)
    stim_input = (stim_input - (stim_input.mean(-1)[:,None]))/stim_input.std(-1)[:,None]
    return stim_input


def Input_pref_ori(theta_stim, theta_pref, sigma_theta=np.deg2rad(30), ampl=1.):
	exp_1=np.exp(2j*theta_stim)
	exp_2=np.exp(2j*theta_pref)
	stimulus_diff=np.arccos(np.dot(exp_1[:,None], np.conjugate(exp_2[None,:])).real)
	return ampl*np.exp(-stimulus_diff**2/(2*sigma_theta)**2)

def generate_activity_prevLayer(network_params):
	Nprev=network_params["num_neurons_previous"]
	nstim=network_params['nstim']

	theta_pref=np.arange(Nprev)/Nprev*np.pi
	alphas=np.arange(nstim)/nstim*np.pi
	tuned_component= Input_pref_ori(theta_stim=alphas, theta_pref=theta_pref)
	tuned_component = (tuned_component - (tuned_component.mean(-1)[:,None]))/tuned_component.std(-1)[:,None]
	return tuned_component

def generate_trials(tuned_comp,
                   Ntrial=10,
                   noise_strength=1,
                   noise_mode='white',
                   sigma_structured_noise=1
                   ):

    rng_start_activity = np.random.default_rng()

    if noise_mode=='white':
        input_rnd = rng_start_activity.normal(size=(Ntrial,)+tuned_comp.shape)
        Inputs = noise_strength*input_rnd + tuned_comp[None,:]

    if noise_mode=='structured':
        N=int(np.sqrt(tuned_comp.shape[-1]))
        shape_wo_neuron=tuned_comp.shape[:-1]
        n_noise_pats = Ntrial*np.prod(shape_wo_neuron)
        additive_noise = get_additive_noise(
            (n_noise_pats,N,N), {}, None,
            noise_level=sigma_structured_noise)

        additive_noise = np.reshape(additive_noise, (Ntrial,)+tuned_comp.shape)

        Inputs = noise_strength*additive_noise + tuned_comp[None,:]
    return Inputs

#if __name__=='__main__':
#    import matplotlib.pyplot as plt
#    nevents=10; index=1
#    N, M = 60,60
#    param1=0.8
#    param2=0.3
#    param3=1.02
#    ntrial=10
#    nstim=4
#    network_params = {'input_shape' 		: np.array([N, M]),
#				  'mode'				: 'short_range',
#				  'noise_type'			: 'postsyn',
#				  'sigmax'				: 1.8,
#				  'sigmax_sd'			: param1*0.15/0.8,
#				  'ecc'					: param1,
#				  'ecc_sd'				: 0.13*param1,
#				  'orientation'			: 0,
#				  'orientation_sd'		: 1.0,
#				  'amplitude'			: 1.,
#				  'inh_factor'			: 2.5,
#				  'input_noise_level'	: param2,
#				  'nonlinearity_rule'	: 'rectification',
#				  'dt'					: 0.15,
#				  'runtime'				: 10,		## number of integration steps
#				  'nonlin_fac'			: param3,
#				  'nstim'			: nstim,
#				  'ntrial'			: ntrial,
#
#				  }
#

#    additive_noise = get_additive_noise((nevents,N,M), network_params, index   )
#    print(additive_noise.shape)
#    print(additive_noise.std(-1))
#    print(additive_noise.mean(-1))
#    #print(abs(additive_noise).max(-1))


#    additive_noise = get_SCM("tunint_random",network_params,index, filename="activity_v0", folder_index_old="0/")
#    print(additive_noise.shape)
#    print(additive_noise.std(-1))
#    print(additive_noise.mean(-1))
#    print(abs(additive_noise).max(-1))

#    stim_input = get_SCM("tuning_random",network_params,folder_index_old="0/")
#    stim_input = stim_input.reshape(-1,60,60)

#    print(stim_input.shape)
#    print(stim_input[0].shape)
#    plt.subplot(141)
#    plt.imshow(stim_input[0],interpolation='nearest',cmap='binary')
#    plt.colorbar()
#    plt.subplot(142)
#    plt.imshow(stim_input[1],interpolation='nearest',cmap='binary')
#    plt.colorbar()
#    plt.subplot(143)
#    plt.imshow(stim_input[2],interpolation='nearest',cmap='binary')
#    plt.colorbar()
#    plt.subplot(144)
#    plt.imshow(stim_input[3],interpolation='nearest',cmap='binary')
#    plt.colorbar()
#    plt.savefig('test_fft2.pdf')
