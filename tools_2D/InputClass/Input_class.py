"#!/usr/bin/env python3"
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:32:33 2023

@author: sigridtragenap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



import sys
from os.path import abspath, sep, pardir
path_parent = abspath('') + sep + pardir + sep + pardir + sep
sys.path.append(path_parent)

from tools_2D.InputClass import StimTypes
from tools_2D.InputClass import ReceptiveField_funcs

import tools_2D.simulate.bn_tools_np as bnt
from tools_2D.simulate import generate_noisy_mh

# from Modular_2D.simulate.tools import generate_noisy_mh
#from Modular_2D.simulate.tools i import mport generate_input_RF
# import Modular_2D.simulate.tools.bn_tools_np as bnt
# import Modular_2D.simulate.tools.brain_network_tf as bn
# from Modular_2D.simulate.tools import connectivity_params
# from Modular_2D.simulate.tools import save_activity

class StimulusClass:
    def __init__(self,
                 numeric_params,
                 index=None):


        self.numeric_params = numeric_params



    def generate_stim_angle(self,
                       stim_params,
                       angle):

        if stim_params['stim_type']=='SinusoidGrating':
            stim = StimTypes.SinuisoidGrating(
                            Size=self.numeric_params["size_stimulated_VF"],
                            A=1,
                            angle_deg=-1*angle,
                            frequency_Kspace=stim_params['frequency_Kspace'],
                             phase=stim_params['phases'],
                             dx=self.numeric_params["dx_visualfield"])

        if stim_params['stim_type']=='Bar':
            #print('Ellipse')
            phases = stim_params['phases']
            stimuli_allphases=[]
            for phase in phases:
                #print(phase)
                stim_singlephase = StimTypes.generate_bar(
                    Size=self.numeric_params["size_stimulated_VF"],
                    pos=phase,
                    angle=angle,
                    bar_width=stim_params['bar_width'],
                    bar_length=stim_params['bar_length'],
                    dx=self.numeric_params["dx_visualfield"],
                    pbc=stim_params['periodic_stimulus'])
                stimuli_allphases.append(stim_singlephase)

            stim = np.asarray(stimuli_allphases)

        if stim_params['stim_type'] == 'BumpyBar':
            phases = stim_params['phases']
            stimuli_allphases = []
            for phase in phases:
                stim_singlephase = StimTypes.generate_bumpy_bar(
                    Size=self.numeric_params["size_stimulated_VF"],
                    pos=phase,
                    angle=angle,
                    bar_width=stim_params['bar_width'],
                    bar_length=stim_params['bar_length'],
                    noise_params=stim_params['noise_params'],
                    dx=self.numeric_params["dx_visualfield"],
                    pbc=stim_params['periodic_stimulus'])
                stimuli_allphases.append(stim_singlephase)
            stim = np.asarray(stimuli_allphases)

        if stim_params['stim_type']=='Ellipse':
            phases = stim_params['phases']
            stimuli_allphases=[]
            for phase in phases:
                stim_singlephase = StimTypes.generateEllipse(
                    Size=self.numeric_params["size_stimulated_VF"],
                    pos=phase,
                    angle=angle,
                    sigma1=stim_params['ellipse_sigma1'],
                    sigma2=stim_params['ellipse_sigma2'],
                    dx=self.numeric_params["dx_visualfield"],
                    pbc=stim_params['periodic_stimulus'])
                stim_singlephase /= np.max(stim_singlephase)
                stimuli_allphases.append(stim_singlephase)
            stim = np.asarray(stimuli_allphases)


        if stim_params['stim_type'] == 'Fullfield':
            N = self.numeric_params["size_stimulated_VF"]
            stim = np.ones((1, N, N))

        if stim_params['stim_type'] == 'Bumps':
            phases = stim_params['phases']
            stimuli_allphases = []
            for phase in phases:
                stim_singlephase = StimTypes.generate_bumps(
                    Size=self.numeric_params["size_stimulated_VF"],
                    pos=phase,
                    angle=angle,
                    spacing=stim_params.get('spacing', 10),
                    sigma=stim_params.get('sigma', None),
                    n_rows=stim_params.get('n_rows', 1),
                    row_spacing=stim_params.get('row_spacing', None),
                    pbc=stim_params.get('periodic_stimulus', True))
                stimuli_allphases.append(stim_singlephase)
            stim = np.asarray(stimuli_allphases)

        if stim_params['stim_type'] == 'MeanderCurve':
            phases = stim_params['phases']
            stimuli_allphases = []
            for phase in phases:
                # We use generate_curved_bar because it handles both curvature AND meandering
                stim_singlephase = StimTypes.generate_curved_bar(
                    Size=self.numeric_params["size_stimulated_VF"],
                    pos=phase,
                    angle=angle,
                    bar_width=stim_params.get('bar_width', 3),
                    curvedness=stim_params.get('curvedness', 0.03), # > 0 means crescent
                    bar_length=stim_params.get('bar_length',
                                               self.numeric_params["size_stimulated_VF"] * 2),
                    meander_amp=stim_params.get('meander_amp', 30.0), # > 0 means wobble
                    meander_freq=stim_params.get('meander_freq', 1.0),
                    dx=self.numeric_params["dx_visualfield"],
                    pbc=stim_params.get('periodic_stimulus', True)
                )
                stimuli_allphases.append(stim_singlephase)

            stim = np.asarray(stimuli_allphases)

        return stim

    def generate_stimuli(self,
                       stim_params):

        stimuli_angles=stim_params['visual_angles']

        stimuli_allangles=[]
        for angle in stimuli_angles:
            stimuli_single = self.generate_stim_angle(
                               stim_params,
                               angle)

            stimuli_allangles.append(stimuli_single)
        return np.asarray(stimuli_allangles)




class InputLayer_evoked:
    def __init__(self,
                receptive_field_params,
                feedforward_connectivity_params,
                numeric_params,
                index=None):
        self.index=index
        self.receptive_field_params = receptive_field_params
        self.feedforward_params = feedforward_connectivity_params
        self.numeric_params = numeric_params

        self._init_receptivefields()
        self._init_feedforward_conn()

        self.init_diffeq_L4()


    def _init_receptivefields(self):
        self.N, self.M=self.feedforward_params["N,M"]
        size_visualspace=self.numeric_params["size_visualspace"] #size of VF that neurons cover
        size_stimulated_VF = self.numeric_params["size_stimulated_VF"] #size of stimulus (angles)
        dx_visualfield = self.numeric_params["dx_visualfield"]  #resolution of stimulus
        dx_grid = size_visualspace/self.N
        ngrid_visualfield = int(size_stimulated_VF/dx_visualfield)


        x_grid = np.arange(-size_visualspace/2.0,size_visualspace/2.,dx_grid)
        X_center, Y_center = np.meshgrid(x_grid, x_grid)
        rng = np.random.default_rng(self.index)


        #set strength of dipoles
        if self.receptive_field_params["dipol_mode"]:
            if self.receptive_field_params["d_gauss_mode"]:
                sigma_mean_d = self.receptive_field_params["sigma_mean_d"]
                sigma_std_d = self.receptive_field_params["sigma_std_d"]
            else:
                dmin = self.receptive_field_params["dmin"]
                dmax = self.receptive_field_params["dmax"]
        else:
            dmin=0
            dmax=0

        #set eccentricity of rf
        if self.receptive_field_params["ellipsoidal_RF"]:
            sigma_mean = self.receptive_field_params["sigma_mean"]
            sigma_std = self.receptive_field_params["sigma_std"]
            sigma_min = self.receptive_field_params["sigma_min"]
            sigma_max = self.receptive_field_params["sigma_max"]
        else:
            sigma_mean = self.receptive_field_params["sigma_mean"]
            sigma_std=0
            sigma_min=0
            sigma_max=np.inf

        #OnOffbalance
        if self.receptive_field_params["onoff_balance"]=='random':
            onoff_balance_std = self.receptive_field_params["onoff_balance_spread"]
            onoffmin = 0.5-0.5*(onoff_balance_std)
            onoffmax = 0.5+0.5*(onoff_balance_std)
            onoff_dominance = rng.uniform(low=onoffmin,
                                          high=onoffmax,
                                         size=(self.N,self.M))

        if self.receptive_field_params["onoff_balance"]=='bimodal':
              proportion_off= self.receptive_field_params["onoff_proportion_off"]
              onoff_identity = rng.uniform(low=0, high=1, size=(self.N,self.M))
              onoff_identity[onoff_identity<proportion_off]=0.
              onoff_identity[onoff_identity>=proportion_off]=1.

              #get half-normal distribution
              onoff_balance_std = self.receptive_field_params["onoff_balance_spread"]
              onoff_dominance_pre =rng.normal(loc=0, scale=onoff_balance_std,
                                           size=(self.N,self.M))

              #mirror values for off-dominated
              onoff_dominance = np.zeros((self.N, self.M,))
              onoff_dominance[onoff_identity==0] = (onoff_identity + np.abs(onoff_dominance_pre))[onoff_identity==0]
              onoff_dominance[onoff_identity==1] = (onoff_identity - np.abs(onoff_dominance_pre))[onoff_identity==1]
              onoff_dominance = np.clip(onoff_dominance, a_min=0.02, a_max=.98)

        elif self.receptive_field_params["onoff_balance"]=='map':
            sigmas_onoff = self.receptive_field_params["onoff_balance_sigmas"]
            onoff_balance_std = self.receptive_field_params["onoff_balance_spread"]

            onoff_dominance = rng.normal(loc=0, scale=onoff_balance_std,
                                         size=(self.N,self.M))
            onoff_dominance_conv=generate_noisy_mh.convolve_with_MH(
                onoff_dominance,N,M,
                sigma1=sigmas_onoff[0],sigma2=sigmas_onoff[1],
                return_real=True,padd=0)
            onoff_dominance_conv -= np.mean(onoff_dominance_conv)
            onoff_dominance_conv /= np.std(onoff_dominance_conv)
            onoff_dominance_conv *= onoff_balance_std
            onoff_dominance_conv += 0.5
            onoff_dominance = np.clip(onoff_dominance_conv, a_min=0, a_max=1)


        #Orientation Dipole
        if self.receptive_field_params["orientation_mode"]=='random':
            orientation_subfields = rng.uniform(low=0, high=360,
                                                size=(self.N,self.M))
        elif self.receptive_field_params["orientation_mode"]=='map':
            orientation_sigma = self.receptive_field_params["orientation_sigma"]
            print("map orientation not yet implemented")




        # same sigmax, sigmax and tilt for on/off
        # varying onoff balance
        # varying distances between subfields (unstructured)
        # dipol orientation random (unstructured)

        center_positions = np.stack([X_center, Y_center], axis=-1)
        if self.receptive_field_params["d_gauss_mode"]:
            offsets_subfields = rng.normal(loc=sigma_mean_d, scale=sigma_std_d, size=(self.N, self.M))
        else:
            offsets_subfields = rng.uniform(low=dmin, high=dmax, size=(self.N, self.M))

        #orientation_subfields = rng.uniform(low=0, high=360, size=(self.N,self.M))
        sigmas_subfields_single = rng.normal(loc=sigma_mean, scale=sigma_std,
                                      size=(self.N,self.M,2))
        sigmas_subfields = np.stack([sigmas_subfields_single, sigmas_subfields_single],
                                  axis=-1)
        sigmas_subfields = np.clip(sigmas_subfields, a_min=sigma_min, a_max=sigma_max)

        #tilt of subfields is identical to sigma_orientation  + 90
        #leads to more 'nice dipoles'
        tilt_subfields = np.stack([orientation_subfields+90, orientation_subfields+90],
                                  axis=-1)

        #create Parameter dicitonary
        rf_param_dict={
            "center_positions" : center_positions,
            "offsets_subfields": offsets_subfields,
            "orientation_subfields": orientation_subfields,
            "sigmas_subfields": sigmas_subfields,
            "tilt_subfields": tilt_subfields,
            "onoff_dominance": onoff_dominance,}
        self.rf_params_instance = rf_param_dict


        Array_receptive_fields = np.zeros((self.N,self.M,ngrid_visualfield,ngrid_visualfield))
        ONOFF_Array_receptive_fields = np.zeros((2,self.N,self.M,ngrid_visualfield,ngrid_visualfield))
        for i in np.arange(self.N):
            for j in np.arange(self.M):
                #print(rf_param_dict["offsets_subfields"][i,j])
                composite_rf, on_rf,off_rf=ReceptiveField_funcs.create_subfieldRF(
                                    rf_param_dict["center_positions"][i,j],
                                    rf_param_dict["offsets_subfields"][i,j],
                                    rf_param_dict["orientation_subfields"][i,j],
                                    rf_param_dict["sigmas_subfields"][i,j,:,0],
                                    rf_param_dict["tilt_subfields"][i,j,0],
                                    rf_param_dict["sigmas_subfields"][i,j,:,1],
                                    rf_param_dict["tilt_subfields"][i,j,1],
                                    rf_param_dict["onoff_dominance"][i,j],
                                    Size=size_stimulated_VF,
                                    dx=dx_visualfield)
                Array_receptive_fields[i,j]=composite_rf
                ONOFF_Array_receptive_fields[0,i,j]=on_rf
                ONOFF_Array_receptive_fields[1,i,j]=off_rf
        self.receptive_fields_array = Array_receptive_fields
        self.ONOFF_receptive_fields_array= ONOFF_Array_receptive_fields


    def _init_feedforward_conn(self, params=None):
        if params is not None:
            self.feedforward_params = params
        eccentricity = self.feedforward_params["eccentricity"]

        binary_conn = self.feedforward_params["mixing_binary"]
        if binary_conn:
            radius_conn = self.feedforward_params["radius_FFconnectivity"]
            sigma_gauss=1
        else:
            sigma_gauss = self.feedforward_params["gauss_sigma"]
            radius_conn=0

        network_params = {
                'mode'                    : 'short_range',
                'noise_type'              : 'None',
                'sigmax'                  : sigma_gauss,
                'sigmax_sd'               : eccentricity*0.15/0.8,
                'ecc'                     : eccentricity,
                'ecc_sd'                  : 0.13*eccentricity,
                'orientation'             : 0,
                'orientation_sd'          : 1.0,
                'amplitude'               : 0.,  ##ensures that only positive weights
                'inh_factor'              : 1.5,
                'pbc'                     : True,
                'represent_binary'        : binary_conn,
                'radius_connectivity'     : radius_conn,
                }

        ##implement sampled gaussian check

        gauss_b,_ = generate_noisy_mh.gaussian_wrap(self.N, self.M,
                                                      network_params,
                          pbc=True,index=self.index,full_output=True,version=0)

        gauss_2d = np.reshape(gauss_b.T, (self.N*self.M,self.N,self.M))
        #Normalize so that total sum is 1
        gauss_2d /= np.sum(gauss_2d, axis=0)[None,:,:]
        if self.feedforward_params["sampling"]:
            K_total = self.feedforward_params["sampling_K"]
            rng=np.random.RandomState(seed=self.index)

            empty_idx=(gauss_2d.sum(0)==0)
            #print(empty_idx.shape)
            gauss_2d=rng.binomial(K_total, gauss_2d)*1.
            gauss_2d[empty_idx.flatten(),empty_idx]=1
            w_flat=gauss_2d.reshape(self.N*self.M, self.N*self.M)
            empty_idx=w_flat.sum(0)==0
            w_flat[empty_idx,empty_idx]=1
            gauss_2d=w_flat.reshape(self.N*self.M, self.N,self.M)
            gauss_2d /= np.sum(gauss_2d, axis=0)[None,:,:]


        #gauss_2d /= np.sum(gauss_2d, axis=0)[None,:,:]
        self.w1D_ff_l4_l2 = gauss_2d
        self.w2D_ff_l4_l2 = gauss_2d.reshape(self.N, self.M, self.N, self.M)

    def init_diffeq_L4(self):
        self.l4_nonlinearity_rule = self.receptive_field_params["l4_nonlinearity"]
        self.layer4_response_mode = self.receptive_field_params["l4_activity_mode"]
        if self.l4_nonlinearity_rule == 'rectification':
            self.layer4_nonlinearity = bnt.nl_rect

        if self.layer4_response_mode == 'instant':
            base_rate = self.receptive_field_params["base_rate"]
            def instant_activation(linear_space,
                                   base_rate=1,
                                   nonlinearity=lambda x:x):

                return base_rate+nonlinearity(linear_space)

            self.layer4_computation = lambda x: instant_activation(x,
                                                                   base_rate,
                                                                   self.layer4_nonlinearity)
        if self.layer4_response_mode == 'membrane':
            print("not yet implemented")


    def calc_L4_activity(self,stim):
        self.layer4_computation_mode = self.receptive_field_params["l4_RF_mode"]
        #einsum is also slow
        # linear_space = np.einsum('nmxy,txy->tnm',
        #                          self.receptive_fields_array,stim)
        #activation = np.sum(Array_receptive_fields[None]*stim[:,None,None,:,:],
        #                    axis=(-1,-2))   #slower variant  xx
        def convolve_stim_RF(receptive_field, stim):
            N,M,X,Y = receptive_field.shape
            receptive_field_tmp = receptive_field.reshape(N*M,X*Y)
            stim_flat = stim.reshape(-1,X*Y)
            response = stim_flat@receptive_field_tmp.T
            linear_space = response.reshape(-1,N,M)
            return linear_space


        if self.layer4_computation_mode == 'Linear_RF':
            linear_space = convolve_stim_RF(self.receptive_fields_array, stim)

        if self.layer4_computation_mode == 'Semilinear_OnOff':
            stim_pos_part = np.clip(stim, a_min=0, a_max=None)
            on_contribution = convolve_stim_RF(
                self.ONOFF_receptive_fields_array[0], stim_pos_part)
            del stim_pos_part
            stim_neg_part = np.clip(stim, a_min=None, a_max=0)
            off_contribution = convolve_stim_RF(
                self.ONOFF_receptive_fields_array[1], stim_neg_part)

            linear_space = on_contribution+off_contribution

        L4_activity = self.layer4_computation(linear_space)
        return L4_activity

    def calc_L23_actvitiy(self,stim):
        L4_activity = self.calc_L4_activity(stim)
        responses_flat = np.reshape(L4_activity, (-1,self.N*self.M))

        Input_to_L23 = np.einsum( 'ti,inm->tnm',
                                 responses_flat, self.w1D_ff_l4_l2,)
        return {'Input_to_L23': Input_to_L23,
                'Layer4_activity': L4_activity}

    def _generate_stim(self,
                       visual_angle=0,
                       frequency_Kspace=1/2,
                       phases=np.linspace(0, 2*np.pi,20)):
        stim = StimTypes.SinuisoidGrating(
                            Size=self.numeric_params["size_stimulated_VF"],
                            A=1,
                            angle=visual_angle,
                             frequency_Kspace=frequency_Kspace,
                             phase=phases,
                             dx=self.numeric_params["dx_visualfield"])
        return stim

    def calc_responses(self,
                       stim_params):
        stimuli_angles = np.asarray(stim_params["stimuli_angles"])
        res_L4_activity=[]
        res_L2_input=[]
        for angle in stimuli_angles:
            stim=self._generate_stim(
                visual_angle=angle,
                frequency_Kspace=stim_params["spatial_frequency"],
                phases=stim_params["stimuli_phases"]
                                    )
            res_dict = self.calc_L23_actvitiy(stim)
            res_L4_activity.append(res_dict["Layer4_activity"])
            res_L2_input.append(res_dict["Input_to_L23"])

        return {'Input_to_L23': np.asarray(res_L2_input),
                'Layer4_activity': np.asarray(res_L4_activity)}



if __name__=='__main__':
    N,M=40,40
    test_L4=False
    test_stimulus=True

    if test_stimulus:
        numeric_params = {
            'size_visualspace'      : N,  #covered by neurons
            'size_stimulated_VF'    : N,  #stimulated in space
            'dx_visualfield'        : 1  #resolution of stimulus
            }
        n_stim=16
        n_phasepoints=10
        stim_params = {
            'stim_type'              :'SinusoidGrating',
            'visual_angles'          : -np.arange(0,360,360/n_stim)+90,
            'phases'                 : np.linspace(0,2*np.pi,n_phasepoints),
            'frequency_Kspace'       : 1,
            }


        # stim_params = {
        #     'stim_type'              :'Bar',
        #     'visual_angles'          : np.arange(0,360,360/n_stim),
        #     'phases'                 : np.linspace(0,1,n_phasepoints),
        #     'bar_width'              : 10,
        #     'bar_length'             : None,
        #     'periodic_stimulus'      :True,
        #     }


        # stim_params = {
        #     'stim_type'              :'Ellipse',
        #     'visual_angles'          : np.arange(0,360,360/n_stim),
        #     'phases'                 : np.linspace(0,1,n_phasepoints),
        #     'ellipse_sigma1'         : 30,
        #     'ellipse_sigma2'         : np.inf,
        #     }
        StimulusGenerator =  StimulusClass(
                         numeric_params)

        stimuli = StimulusGenerator.generate_stimuli(stim_params)
        #shape (n_stimuli, n_phasepoint, N,M)

        plt.hist(stimuli.flatten())
        plt.show()

        fig, axs = plt.subplots(nrows=16, ncols=n_phasepoints,
                                figsize=(1*n_phasepoints,3))

        for istim in range(16):
            for iphase in range(n_phasepoints):
                axs[istim, iphase].imshow(stimuli[istim, iphase])
                axs[istim, iphase].axis('off')

        plt.show()


    if test_L4:
        size_stimulated_VF=50
        dx_visualfield=0.1
        receptive_field_params={
            'l4_nonlinearity'       : 'rectification',
            'l4_activity_mode'      : 'instant',
            'base_rate'             : 0,
            'onoff_balance'         : 'random',
            'onoff_balance_spread'  : 1,
            'onoff_balance_sigma'   : [1,5],
            'orientation_mode'      : 'random',
            'orientation_sigma'     : 1,
            'ellipsoidal_RF'        : True,
            'sigma_mean'            : 2,
            'sigma_std'             : 1,
            'sigma_min'             : 0.5,
            'sigma_max'             : 5,
            'dipol_mode'            : True,
            'dmin'                  : 0,
            'dmax'                  : 1,
            "d_gauss_mode"          :False,
            'l4_RF_mode'            :'Linear_RF'
            }


        feedforward_connectivity_params={
            'N,M'                   : (N,M),
            'eccentricity'          : 0,
            'mixing_binary'         : True,
            'radius_FFconnectivity' : 1,
            'gauss_sigma'           : 2,
            'sampling'              : False,
            'sampling_K'            : 10,
            }


        numeric_params = {
            'size_visualspace'      : 20,  #covered by neurons
            'size_stimulated_VF'    : size_stimulated_VF,  #stimulated
            'dx_visualfield'        : dx_visualfield  #resolution of stimulus
            }

        InputNet =  InputLayer_evoked(receptive_field_params,
                         feedforward_connectivity_params,
                         numeric_params)

        #
        stim_angle=0
        stim = StimTypes.SinuisoidGrating(size_stimulated_VF, A=1,
                                                  angle=stim_angle,
                             frequency_Kspace=1/2,
                             phase=np.linspace(0,2*np.pi,20), Pos=[0,0],
                             dx=dx_visualfield)

        activity = InputNet.calc_L23_actvitiy(stim)

        #
        n_stim=16
        stimuli = np.arange(0,360,360/n_stim)
        stim_phases=np.linspace(0,2*np.pi,20)
        K=1/1.
        stim_params={
            'stimuli_angles': np.deg2rad(stimuli),
            'stimuli_phases': stim_phases,
            'spatial_frequency': K}

        res=InputNet.calc_responses(stim_params)

        idx_i, idx_j = 20,20

        fig=plt.figure()
        layer_4_act = res["Layer4_activity"][:,:,idx_i,idx_j]
        for istim in [0,2,4,6]:
            plt.plot(layer_4_act[istim])
        plt.show()

        fig=plt.figure()
        layer_2_in = res["Input_to_L23"][:,:,idx_i,idx_j]
        for istim in [0,2,4,6]:
            plt.plot(layer_2_in[istim])
        plt.show()

        def get_F1_F0(activity):
            num_stim, n_phase = activity.shape[:2]
            fft_coeffs = np.fft.fft(activity, axis=1)
            f1 = np.abs(fft_coeffs[:,1])
            f0 = np.mean(activity, axis=1)
            return f0,f1

        f0,f1 = get_F1_F0(res["Layer4_activity"])
        plt.scatter(f0,f1)
        f0,f1 = get_F1_F0(res["Input_to_L23"])
        plt.scatter(f0,f1)
        plt.xlabel("F0")
        plt.ylabel("F1")
        plt.show()

    #%%
        plt.imshow(InputNet.w2D_ff_l4_l2[:,:,15,15])
        plt.colorbar()
        plt.show()



