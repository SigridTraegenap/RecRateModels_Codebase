#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:51:49 2023

@author: sigridtragenap
"""

import numpy as np


def get_pos_onoff(d, angle_onoff_dipol,
                           pos_center = (0,0)):
    angle = np.deg2rad(angle_onoff_dipol)
    x0,y0 = pos_center
    x = d/2. * np.cos(angle)
    y = d/2. * np.sin(angle)
    on_pos = x+x0,y+y0
    off_pos = -1*x+x0, -1*y+y0
    return on_pos, off_pos

def create_subfieldRF(pos_center,
                      d,
                      angle_onoff_dipol,
                      on_sigmas,
                      on_theta,
                      off_sigmas,
                      off_theta,
                      alpha_onoffdominance,
                      Size=100,
                      dx=0.1):

    on_pos, off_pos = get_pos_onoff(d, angle_onoff_dipol,
                                    pos_center)

    #print(on_pos, off_pos)

    RF_Field_on = getEllipsoid(Size, Pos=on_pos,
                               angle=np.deg2rad(on_theta), sigmas=on_sigmas,
                               dx=dx)
    RF_Field_on *=alpha_onoffdominance


    RF_Field_off = getEllipsoid(Size, Pos=off_pos,
                               angle=np.deg2rad(off_theta), sigmas=off_sigmas,
                               dx=dx)
    RF_Field_off *=(alpha_onoffdominance-1)
    RF_composite = RF_Field_on + RF_Field_off

    # x = np.arange(-Size/2.0,Size/2.,dx)
    # X, Y = np.meshgrid(x, x)
    # plt.pcolormesh(X,Y,RF_composite, norm=mpl.colors.CenteredNorm(),
    #                cmap='RdBu_r')
    # plt.axis('square')
    # plt.colorbar()
    # plt.show()


    return RF_composite, RF_Field_on, RF_Field_off


#@title Input definitions
def SinuisoidGrating(Size, A=1, angle=0,
                     frequency_Kspace=10,
                     phase=0, Pos=[0,0],
                     dx=1):
    phase = np.asarray(phase)
    # Generate Sinusoid grating
    #Size
    x = np.arange(-Size/2.0,Size/2.,dx)
    X, Y = np.meshgrid(x, x)

    if len(phase.shape)>0:
        grating=[]

        for phi in phase:
            grating.append(A*np.cos(
               frequency_Kspace*X*np.cos(angle) +
               frequency_Kspace*Y*np.sin(angle) -
               phi))
        grating = np.asarray(grating)

    if len(phase.shape)==0:
        grating = A*np.cos(
           frequency_Kspace*X*np.cos(angle) +
           frequency_Kspace*Y*np.sin(angle) -
           phase)
    return grating

def getGabor(Size, angle=0, frequency_Kspace=10, sigmas=[10,10], phase=0,
             Pos=(0,0),dx=1):

    x = np.arange(-Size/2.0,Size/2.,dx)
    X, Y = np.meshgrid(x, x)

    gauss = 1./(2*np.pi*sigmas[0]*sigmas[1])*np.exp(- X**2/(2*sigmas[0]**2)
                                                    -Y**2/(2*sigmas[1]**2 ))
    cosine = np.cos(
        frequency_Kspace*X*np.cos(angle) +
        frequency_Kspace*Y*np.sin(angle) -
        phase)

    gabor = gauss*cosine
    return gabor/np.abs(gabor).sum()

def getEllipsoid(Size, Pos=(0,0), angle=0, sigmas=(10,10), dx=1):
    x = np.arange(-Size/2.0,Size/2.,dx)
    X, Y = np.meshgrid(x, x)

    a = np.cos(angle)**2/(2*sigmas[0])+np.sin(angle)**2/(2*sigmas[1])
    b = -np.sin(2*angle)/(4*sigmas[0])+np.sin(2*angle)/(4*sigmas[1])
    c = np.sin(angle)**2/(2*sigmas[0])+np.cos(angle)**2/(2*sigmas[1])

    gauss = np.exp(- (a*(X-Pos[0])**2 +
                      c*(Y-Pos[1])**2 +
                    2*b*(X-Pos[0])*(Y-Pos[1])
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


def generate_input(N,M,
                   sigma_av=10,
                   sigma_spread=2,
                   Kstim=1/6,
                   Nstim=16):

    Size=200
    n_neurons=N*M
    delta_stim=360/Nstim
    stims=np.arange(0,360,delta_stim)
    inputs_all_neurons=np.empty((n_neurons,len(stims),))*np.nan

    for ineur in range(n_neurons):
      #sample properties of receptive field
      rng = np.random.default_rng(ineur)
      pref_ori=rng.random()*180
      sigmax=rng.normal()*sigma_spread+sigma_av
      sigmay=rng.normal()*sigma_spread+sigma_av
      rf_ellipse=getEllipsoid(Size=Size, angle=np.deg2rad(pref_ori),
                      sigmas=[sigmax, sigmay])

      #create tuning curve, activity
      response = get_responses(rf_ellipse,
                        stims=stims,
                        Kstim=Kstim)
      inputs_all_neurons[ineur]=response[:,0]
    return inputs_all_neurons.T



def generate_input_Gabor(N,M,
                   sigma_av=10,
                   sigma_spread=2,
                   Kstim=1/6,
                   Nstim=16,
                   K_rf_ac=6):

    Size=200
    n_neurons=N*M
    delta_stim=360/Nstim
    stims=np.arange(0,360,delta_stim)
    inputs_all_neurons=np.empty((n_neurons,len(stims),))*np.nan

    for ineur in range(n_neurons):
      #sample properties of receptive field
      rng = np.random.default_rng(ineur)
      pref_ori=rng.random()*180
      sigmax=rng.normal()*sigma_spread+sigma_av
      sigmay=rng.normal()*sigma_spread+sigma_av
      phase_rf=rng.uniform()
      K_rf = np.clip(1./(K_rf_ac+rng.normal()), a_min=0.001, a_max=None)
      rf_ellipse=getGabor(Size=Size, angle=np.deg2rad(pref_ori),
                       frequency_Kspace=K_rf,
                       sigmas=[sigmax, sigmay],
                       phase=phase_rf,
                          Pos=(0,0),)
      #create tuning curve, activity
      response = get_responses(rf_ellipse,
                        stims=stims,
                        Kstim=Kstim)
      inputs_all_neurons[ineur]=response[:,0]
    return inputs_all_neurons.T


def generate_trials_old(N,M,
                    sigma_av=10,
                    sigma_spread=2,
                    Kstim=1/6,
                    Nstim=16,
                    Ntrial=10,
                    noise_strength=1,
                    mode="Ellipse"):

    if mode=='Ellipse':
        tuned_comp=generate_input(N,M,
                            sigma_av=sigma_av,
                            sigma_spread=sigma_spread,
                            Kstim=Kstim,
                            Nstim=Nstim)
    if mode=="Gabor":
        tuned_comp=generate_input_Gabor(N,M,
                            sigma_av=sigma_av,
                            sigma_spread=sigma_spread,
                            Kstim=Kstim,
                            Nstim=Nstim)

    rng_start_activity = np.random.default_rng()
    input_rnd = rng_start_activity.random(size=[Ntrial,Nstim, N*M])
    Inputs = noise_strength*input_rnd + tuned_comp[None,:]
    return Inputs

def generate_trials(tuned_comp,
                   Ntrial=10,
                   noise_strength=1,):

    rng_start_activity = np.random.default_rng()
    input_rnd = rng_start_activity.normal(size=(Ntrial,)+tuned_comp.shape)
    Inputs = noise_strength*input_rnd + tuned_comp[None,:]
    return Inputs
