#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:22:57 2023

@author: sigridtragenap
"""
import numpy as np
import h5py
from matplotlib.colors import hsv_to_rgb
from skimage.transform import resize
from numpy.random import default_rng
from scipy import linalg


def get_FF_mask(r1,r2,N,M):
    coord_x,coord_y= np.meshgrid(np.arange(N),np.arange(M))
    deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
    deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]
    absdeltax = np.abs(deltax)
    absdeltay = np.abs(deltay)
    idxx = np.argmin([absdeltax, N-absdeltax],axis=0)
    idxy = np.argmin([absdeltay, M-absdeltay],axis=0)

    deltax = deltax*(1-idxx) + np.sign(deltax)*(absdeltax-N)*idxx
    deltay = deltay*(1-idxy) + np.sign(deltay)*(absdeltay-M)*idxy

    delta = np.sqrt((deltax)**2+(deltay)**2)

    mask = np.ones_like(delta)
    mask[delta<r1]=0.
    mask[delta>r2]=0.

    mask = mask.reshape(N*M,N*M)
    return mask.reshape(N*M,N*M)

def trial_to_trial_correlation(data_evoked, n_stim, n_ntrial):
    res=[]
    for istim in range(n_stim):
        res.append(np.nanmean(np.corrcoef(data_evoked[:,istim])[np.triu_indices(n_ntrial, k=1)]))
    return np.asarray(res)

def resize_stack(stack, scale=2, **kwargs):
    if len(stack.shape)>2:
        result=[]
        for i in range(stack.shape[0]):
            result.append(resize_stack(stack[i],scale, **kwargs))
        return np.asarray(result)

    w, h = stack.shape
    shape_new =(w//scale,h//scale)
    frame_x=resize(stack, output_shape=shape_new,
                    mode='reflect', order=1)
    return frame_x

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
    opm = np.nansum(tuning*angles[:,None], axis=0)/(np.nansum(np.abs(tuning), axis=0)+1e-6)
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


def dimensionality(eigvals):
    return np.sum(eigvals)**2/np.sum(eigvals**2)


def construct_mat(eigvals, eigvecs):
    #eigvecs.shape  neurons, number eigvecs
    assert len(eigvals)==eigvecs.shape[1]
    n_supply = eigvecs.shape[1]
    n_dim = eigvecs.shape[0]
    res_W = np.zeros((n_dim, n_dim))
    for i in range(n_supply):
        eigenvector = eigvecs[:,i]
        res_W += eigvals[i]* (eigenvector[:,None] @ eigenvector[None,:])
    return res_W

def construct_onb(n_dim, seed=None):
    #construct random, symmetric matrix
    rng = default_rng(seed=seed)
    A= rng.standard_normal(size=(n_dim, n_dim))
    A_sym = A + A.T
    assert np.allclose(A_sym, A_sym.T)
    #calculate eigenvectors
    #because A is symmetric, these will form an orthonormal basis
    _, eigvecs = linalg.eigh(A_sym)
    return eigvecs

def ev_explained_variance(sigma, eigvecs):
    #eigvecs.shape  neurons, number eigvecs
    return np.diag(eigvecs.T @ sigma @ eigvecs)

def construct_transformation_sigma(broadness=0.2, n_dim=200, ONB=None):
    if ONB is None:  #otherwise pass specific basis vectors
        ONB = construct_onb(n_dim)
    x=np.arange(n_dim)
    eigvals_broad = np.exp(-broadness*x)
    eigvals_broad *= n_dim/eigvals_broad.sum()
    #alternative: construct sigma, take sqrtm
    return construct_mat(np.sqrt(eigvals_broad), ONB)
    #take the real part because complex results are numerical errors
    #sigma onoy has positive eigenvalues per construction

def sample_from_normal(cov, mean=None, n_samples=1000, seed=12345):
    if mean is None:
        mean = np.zeros(cov.shape[0])
    rng_state= np.random.default_rng(seed)
    sampled_vectors = rng_state.multivariate_normal(mean, cov, n_samples)
    return sampled_vectors

def create_n_dim_gauss(K, base_patterns=None, n_neurons=200):
    if base_patterns is None:
        base_patterns = construct_onb(n_neurons)
    eigvals_broad = np.exp(-np.arange(n_neurons)/(0.5*K))
    eigvals_broad *= n_neurons/eigvals_broad.sum()
    C_in = construct_mat(eigvals_broad, base_patterns)
    return C_in

def trial_to_trial_correlation(data_evoked, n_stim, n_ntrial):
    res=[]
    for istim in range(n_stim):
        res.append(np.nanmean(np.corrcoef(data_evoked[:,istim])[np.triu_indices(n_ntrial, k=1)]))
    return np.asarray(res)

def closest_matching_pats(patternsA, patternsB):
    n_a = patternsA.shape[0]
    C_all = np.corrcoef(patternsA, patternsB)[:n_a, n_a:]
    #C_all[np.diag_indices()]
    max_sim = np.max(C_all, axis=1)
    return max_sim

def selec_from_tuning(tuning, normkey=True):
    num_stim, num_cells = tuning.shape
    if num_stim==9:
        tuning = tuning[:8]
        return selec_from_tuning(tuning)
    if num_stim==17:
        tuning = tuning[:16]
        return selec_from_tuning(tuning)
    angles = np.arange(num_stim)*np.pi/float(num_stim)
    angles = np.exp(2j*angles)
    #norm=np.nansum(tuning, axis=0)
    norm=np.nansum(np.abs(tuning), axis=0)
    if normkey==False: norm=1
    opm = np.nansum(tuning*angles[:,None], axis=0)/norm
    #/norm
    return np.abs(opm)

def _np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def shuffled_signalcorr(dummy_TrialFrameNeur):
    num_frames, num_neur=dummy_TrialFrameNeur.shape
    if num_frames%2>0:
        return shuffled_signalcorr(dummy_TrialFrameNeur[:-1])
    shuffled_corr=[] = []
    N_half = num_frames//2
    for ineur in range(num_neur):
        D1 = dummy_TrialFrameNeur[:N_half,ineur]
        D2 = dummy_TrialFrameNeur[N_half:]
        res_12 = _np_pearson_cor(D1, D2)[0]

        del D1, D2

        D1 = dummy_TrialFrameNeur[N_half:,ineur]
        D2 = dummy_TrialFrameNeur[:N_half]

        res_21 = _np_pearson_cor(D1, D2)[0]
        shuffled_corr.append(0.5*(res_12+res_21))

        del D1, D2

    shuffled_corr = np.asarray(shuffled_corr)
    return shuffled_corr

