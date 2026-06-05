#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:29:25 2024

@author: sigridtragenap
"""


import numpy as np
import scipy.stats as stats
import help_funs as hf

import scipy.stats as stats



def dprime_preferred(trial_data):
    num_trial,num_stim, num_cells = trial_data.shape

    orth_shift=int(num_stim/4)
    pref_idx = np.argmax(trial_data.mean(0)[:int(num_stim/2)], axis=0)
    act_preferred = trial_data[:,pref_idx,np.arange(num_cells)]
    act_orth = trial_data[:,pref_idx+orth_shift,np.arange(num_cells)]

    mean1 = act_preferred.mean(0)
    mean2 = act_orth.mean(0)
    var1 = act_preferred.var(0)
    var2 = act_orth.var(0)
    # std1 = act_preferred.std(0)
    # std2 = act_orth.std(0)
    dprime_pref = np.abs(mean1-mean2)/np.sqrt(0.5*(var1+var2))
    return dprime_pref

def pairwise_correlation(A, B):
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)

    v1_m = np.ma.array(am, mask=np.isnan(am))
    v2_m = np.ma.array(bm, mask=np.isnan(bm))

    dot_prod=np.ma.dot(v1_m.T, v2_m)

    return dot_prod /  (np.sqrt(
        np.nansum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.nansum(bm**2, axis=0, keepdims=True)))


def pairwise_covariance(A, B):
    n=A.shape[0]
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)

    dot_prod=am.T@bm

    return dot_prod/(n-1)


def pairwise_correlation_wom(A, B):
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm /  (np.sqrt(
        np.sum(am**2, axis=0,
                keepdims=True)).T * np.sqrt(
        np.sum(bm**2, axis=0, keepdims=True)))

def flatten_zscore_act(activity):
    ntrial, nstim, Ncell = activity.shape
    activity = activity.reshape(ntrial*nstim, Ncell)

    #activity -= np.nanmean(activity, axis=1)[:,None]
    activity /= np.nanstd(activity, axis=1)[:,None]
    return activity

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


#Pooled activity plots
def resort_data(data):
    #expected shape: neur, stimuli, trials
    nstimhalf=int(data.shape[1]/2)
    data_resorted = np.empty_like(data)*np.nan
    for ineur in range(data.shape[0]):
        data_single = data[ineur]
        max_idx = np.argmax(np.nanmean(data_single, axis=-1))
        data_roll1 = np.roll(data_single, shift=-max_idx+nstimhalf, axis=0)
        data_resorted[ineur]=data_roll1
    return data_resorted

def dprime_preferred_old(trial_data):
    num_trial,num_stim, num_cells = trial_data.shape

    trial_data_resort = resort_data(trial_data.T).T

    orth_shift=int(num_stim/4)
    pref_idx=int(num_stim/2)
   # pref_idx = np.argmax(trial_data.mean(0)[:int(n_stim/2)], axis=0)
    # act_preferred = trial_data[:,pref_idx,np.arange(num_cells)]
    # act_orth = activity[:,pref_idx+orth_shift,np.arange(num_cells)]

    mean1 = trial_data_resort[:,pref_idx].mean(0)
    mean2 = trial_data_resort[:,orth_shift].mean(0)
    var1 = trial_data_resort[:,pref_idx].std(0)
    var2 = trial_data_resort[:,orth_shift].mean(0).std(0)
    dprime_pref = np.abs(mean1-mean2)/np.sqrt(0.5*(var1+var2))
    return dprime_pref

def stim_indepdent_variance(trial_data):
    ntrial, nstim, ncell = trial_data.shape

    activity_variances = np.var(trial_data, axis=(0,1))
    #print(trial_data.flatten()[:5])
    tuning = trial_data.mean(0)

    activity_noise = trial_data - tuning[None,:]
    noise_variances = np.var(activity_noise, axis=(0,1))
    stim_unexplained_var = noise_variances/activity_variances
    return stim_unexplained_var


def get_corrs_L4_L23(activity_L23, activity_L4,
                     W_mask,
                     num_trials=10, num_stim=10,
                     N=60,M=60):
    #L23 to itself
    activity_L23_flat = activity_L23.reshape(num_trials, num_stim, -1)
    activity_L23_zs = activity_L23_flat - np.nanmean(activity_L23_flat, axis=(0,1))[None,None,:]
    L23_zscore = flatten_zscore_act(activity_L23_zs)
    L23_zscore_stim = L23_zscore.reshape(num_trials, num_stim, -1)


    res_corrL23L23 = pairwise_correlation(L23_zscore, L23_zscore)
    res_corrL23L23[~(W_mask>0)]=0.
    associated_network_act = L23_zscore_stim@res_corrL23L23

    trialflat_data_network = associated_network_act.reshape(num_trials*num_stim,N*M)
    trialflat_data_SU = L23_zscore_stim.reshape(num_trials*num_stim,N*M)
    tuning_network = associated_network_act.mean(0)
    tuning_SU = L23_zscore_stim.mean(0)
    residual_network = (associated_network_act-
                        tuning_network[None,:]).reshape(num_trials*num_stim,N*M)
    residual_SU = (L23_zscore_stim-
                        tuning_SU[None,:]).reshape(num_trials*num_stim,N*M)


    stim_independentvat_L23 = stim_indepdent_variance(associated_network_act)
    #print(associated_network_act.flatten()[:5])
    total_corr_L23 = get_rankcorr_between(trialflat_data_network, trialflat_data_SU)
    tuning_corr_L23 = get_rankcorr_between(tuning_network, tuning_SU)
    res_corr_L23  = get_rankcorr_between(residual_network, residual_SU)



    #L4 to L23
    activity_L4_flat = activity_L4.reshape(num_trials, num_stim, -1)
    activity_L4_zs = activity_L4_flat - np.nanmean(activity_L4_flat, axis=(0,1))[None,None,:]
    L4_zscore = flatten_zscore_act(activity_L4_zs)
    L4_zscore_stim = L4_zscore.reshape(num_trials, num_stim, -1)

    res_corrL4L23 = pairwise_correlation(L4_zscore, L23_zscore)
    res_corrL4L23[~(W_mask>0)]=0.
    associated_network_act = L23_zscore_stim@res_corrL4L23.T

    trialflat_data_network = associated_network_act.reshape(num_trials*num_stim,N*M)
    trialflat_data_SU = L4_zscore_stim.reshape(num_trials*num_stim,N*M)
    tuning_network = associated_network_act.mean(0)
    tuning_SU = L4_zscore_stim.mean(0)
    residual_network = (associated_network_act-
                        tuning_network[None,:]).reshape(num_trials*num_stim,N*M)
    residual_SU = (L4_zscore_stim-
                        tuning_SU[None,:]).reshape(num_trials*num_stim,N*M)

    #print(associated_network_act.flatten()[:5])
    stim_independentvat_L4 = stim_indepdent_variance(associated_network_act)

    total_corr_L4 = get_rankcorr_between(trialflat_data_network, trialflat_data_SU)
    tuning_corr_L4 = get_rankcorr_between(tuning_network, tuning_SU)
    res_corr_L4  = get_rankcorr_between(residual_network, residual_SU)

    return [total_corr_L23, tuning_corr_L23, res_corr_L23, stim_independentvat_L23], [
        total_corr_L4, tuning_corr_L4, res_corr_L4,stim_independentvat_L4]


def get_rankcorr_between(act_A, act_B):
    Ndatapoint, Ncell = act_A.shape
    assert act_A.shape==act_B.shape

    res_sp=stats.spearmanr(act_A, act_B)
    rank_corr = res_sp.statistic[:Ncell,Ncell:]
    return np.diag(rank_corr)


def get_pearsoncorr_between(act_A, act_B):
    corr = hf.np_pearson_cor(act_A, act_B)
    return np.diag(corr)

def get_corrpats(activity_L23, activity_L4,
                 W_mask=None,
                     num_trials=10, num_stim=16,
                     N=60,M=60):


    activity_L23 -= np.nanmean(activity_L23, axis=(0,1))[None,None,:]
    activity_L4 -= np.nanmean(activity_L4, axis=(0,1))[None,None,:]
    L23_zscore = flatten_zscore_act(activity_L23)
    L4_zscore = flatten_zscore_act(activity_L4)

    #get corr L4 L23
    res_corrL4L23 = pairwise_correlation(L4_zscore, L23_zscore)
    rng = np.random.default_rng(1804)
    L23_TrialShuffled = rng.permutation(L23_zscore, axis=0)
    L4_TrialShuffled = rng.permutation(L4_zscore, axis=0)
    res_corr_TrialShuffled = pairwise_correlation(L4_TrialShuffled, L23_TrialShuffled)

    res_corr_zscore = res_corrL4L23.copy()
    res_corr_zscore  -= np.nanmean(res_corr_TrialShuffled)
    res_corr_zscore  /= np.nanstd(res_corr_TrialShuffled)
    res_corrL4L23 = np.copy(res_corr_zscore)

    res_corr_TrialShuffled  -= np.nanmean(res_corr_TrialShuffled)
    res_corr_TrialShuffled  /= np.nanstd(res_corr_TrialShuffled)
    #print("chance corr trial-shuffled:", np.mean(np.abs(res_corr_TrialShuffled)))

    #get corr L23 L4
    res_corrL23L23 = pairwise_correlation(L23_zscore, L23_zscore)
    rng = np.random.default_rng(1804)
    L23_TrialShuffled1 = rng.permutation(L23_zscore, axis=0)
    L23_TrialShuffled2 = rng.permutation(L23_zscore, axis=0)
    res_corr_TrialShuffled = pairwise_correlation(L23_TrialShuffled1, L23_TrialShuffled2)

    res_corr_zscore = res_corrL23L23.copy()
    res_corr_zscore  -= np.nanmean(res_corr_TrialShuffled)
    res_corr_zscore  /= np.nanstd(res_corr_TrialShuffled)
    res_corrL23L23 = np.copy(res_corr_zscore)

    return res_corrL23L23, res_corrL4L23

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
