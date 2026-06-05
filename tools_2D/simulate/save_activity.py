#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:11:11 2024

@author: sigridtragenap
"""

import numpy as np
import os
import h5py
from .global_params import global_path

def save_params(network_params, version, folder_index, filename="Parameters"):
    filepath = global_path

    #Network settings
    full_name = filepath + '{}_v{}.hdf5'.format(filename,version)
    f = h5py.File(full_name,'a')
    for key in network_params.keys():
        #print(key)
        if key in ('activity',):
            continue
        f.create_dataset(folder_index + key, data=network_params[key])
    f.close()
    print('f Params close')

def save_activity(activity, network_params, w_rec, version, folder_index, save_wrec=False):
    filepath = global_path

    #activity
    full_name = filepath + 'activity_v{}.hdf5'.format(version)
    f = h5py.File(full_name,'a')
    f.create_dataset(folder_index + 'activity', data=activity)
    f.create_dataset(folder_index + 'shape', data=activity.shape)
    f.close()
    print('f activity close')

    #wrec
    if save_wrec:
        full_name = filepath + 'connectivity_v{}.hdf5'.format(version)
        f = h5py.File(full_name,'a')
        f.create_dataset(folder_index + "w_rec", data=w_rec)
        f.create_dataset(folder_index + "eigenvals", data=network_params["eigenvals"])
        f.close()
        print('f connectivity close')

    #Network settings
    full_name = filepath + 'NetworkParams_v{}.hdf5'.format(version)
    f = h5py.File(full_name,'a')
    for key in network_params.keys():
        #print(key)
        if key in ('activity',):
            continue
        f.create_dataset(folder_index + key, data=network_params[key])
    f.close()
    print('f Params close')

def dict_clean(items):
    result = {}
    for key, value in items.items():
        if value is None:
            value = 'None'
        result[key] = value
    return result

def save_Layeractivity(L23_Activity,
                       L23_Inputs,
                       L4_Activity=None,
                       network_params={}, w_rec=None,
                       save_key='',
                       folder_index=0, save_wrec=False):

    filepath = global_path
    #print(filepath)

    #L23activity
    full_name = filepath + 'activity_v{}.hdf5'.format(save_key)
    f = h5py.File(full_name,'a')
    f.create_dataset(folder_index + 'activity', data=L23_Activity)
    f.create_dataset(folder_index + 'shape', data=L23_Activity.shape)
    f.close()


    #L23activity
    full_name = filepath + 'L23_Inputs_v{}.hdf5'.format(save_key)
    f = h5py.File(full_name,'a')
    f.create_dataset(folder_index + 'activity', data=L23_Inputs)
    f.create_dataset(folder_index + 'shape', data=L23_Inputs.shape)
    f.close()

    if L4_Activity is not None:
        #L4activity
        full_name = filepath + 'L4_activity_v{}.hdf5'.format(save_key)
        f = h5py.File(full_name,'a')
        f.create_dataset(folder_index + 'activity', data=L4_Activity)
        f.create_dataset(folder_index + 'shape', data=L4_Activity.shape)
        f.close()
    #print('f activity close')

    #wrec
    if save_wrec:
        full_name = filepath + 'connectivity_v{}.hdf5'.format(save_key)
        f = h5py.File(full_name,'a')
        f.create_dataset(folder_index + "w_rec", data=w_rec)
        f.create_dataset(folder_index + "eigenvals", data=network_params["eigenvals"])
        f.close()
        #print('f connectivity close')

    #Network settings
    full_name = filepath + 'NetworkParams_v{}.hdf5'.format(save_key)
    f = h5py.File(full_name,'a')
    network_params_clean = dict_clean(network_params)
    for key in network_params_clean.keys():
        #print(key)
        if key in ('activity',):
            continue
        f.create_dataset(folder_index + key, data=network_params_clean[key])
    f.close()
    print(folder_index, 'f Params close')


def gimme_index(filename,dim=2):
	filepath = global_path

	if not os.path.exists(filepath):
		print("creating {}".format(filepath))
		os.mkdir(filepath)
	full_name = filepath + filename
	print('Save under: {}'.format(full_name))
	try:
		f = h5py.File(full_name,'a')
		indices = [int(item) for item in sorted(f.keys())]
		max_index = np.max(indices)
		f.create_group('{}'.format(max_index+1))
		f.close()
	except Exception as e:
		print("Exception",e)
		max_index = -1
	return max_index + 1


import scipy.io as sio

def get_current_AnalysisDict(filename):
    try:
        AnalysisDict=sio.loadmat(global_path+filename, simplify_cells=True)
        print('existing analyses', AnalysisDict.keys())
    except Exception as e:
        print("Exception",e)
        AnalysisDict={}
    return AnalysisDict

