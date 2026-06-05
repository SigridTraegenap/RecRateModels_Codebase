#!/usr/bin/env python

'''
Load and visualise results saved by loop_reload_Amodes_250925.py.

HDF5 files are written to  ../res_tf2_2d/  (one level up from this folder).
Run this script from the Example_Networks/ folder:

    cd Example_Networks
    conda run -n tf_pyth python load_results.py
'''

import sys
from os.path import abspath, sep, pardir

# Add repo root to path so tools_2D is importable
path_parent = abspath('') + sep + pardir + sep
sys.path.append(path_parent)

import numpy as np
import h5py
import matplotlib.pyplot as plt

from tools_2D.simulate.global_params import global_path

# =============================================================================
# CONFIGURATION — set these to match the run you want to inspect
# =============================================================================

Save_key    = 'loop_reload_25-12-12_N60_linear'   # must match Save_key used when running
folder_index = 0    # integer group index written by gimme_index (0, 1, 2, …)

N, M = 60, 60      # grid size (must match the run)

# =============================================================================
# LOAD ACTIVITY
# =============================================================================

activity_file = global_path + 'activity_v{}.hdf5'.format(Save_key)
print('Loading:', activity_file)

with h5py.File(activity_file, 'r') as f:
    print('Available groups:', list(f.keys()))
    activity = f[str(folder_index)]['activity'][:]   # read into numpy array

# Saved shape: (n_trials, n_stim, n_timepoints, n_pop, N, M)
print('Activity shape:', activity.shape)
n_trials, n_stim, n_tp, n_pop, _, _ = activity.shape

# =============================================================================
# LOAD NETWORK PARAMETERS
# =============================================================================

params_file = global_path + 'NetworkParams_v{}.hdf5'.format(Save_key)
print('\nLoading:', params_file)

with h5py.File(params_file, 'r') as f:
    group = f[str(folder_index)]
    print('Saved parameters:', list(group.keys()))
    # Example: read a few key params
    nonlin_fac = group['nonlin_fac'][()]
    dt         = group['dt'][()]
    runtime    = group['runtime'][()]

print('gamma (nonlin_fac):', nonlin_fac)
print('dt:', dt, '  runtime steps:', runtime)

# =============================================================================
# VISUALISE: last timepoint of a single activity pattern
# =============================================================================

# Pick one trial and one stimulus
trial_idx = 0
stim_idx  = 0
tp_idx    = -1   # last saved timepoint

pattern = activity[trial_idx, stim_idx, tp_idx, 0]   # shape (N, M)

plt.figure(figsize=(4, 4))
plt.imshow(pattern, interpolation='nearest', cmap='binary_r')
plt.colorbar(label='activity')
plt.title('Trial {} | Stim {} | t={}'.format(trial_idx, stim_idx, tp_idx))
plt.tight_layout()
plt.savefig('example_pattern.pdf')
plt.show()

# =============================================================================
# VISUALISE: seed-point correlation map
# =============================================================================
# Flatten all trials and stimuli, compute the N*M × N*M correlation matrix,
# then pick the centre neuron as seed.

flat = activity[:, :, -1, 0].reshape(-1, N * M)   # (n_trials*n_stim, N*M)
corr_matrix = np.corrcoef(flat.T)                  # (N*M, N*M)

# Reshape to (N, M, N, M) and select centre pixel as seed
corr_4d   = corr_matrix.reshape(N, M, N, M)
corr_map  = corr_4d[N // 2, M // 2]               # (N, M)

plt.figure(figsize=(4, 4))
plt.imshow(corr_map, interpolation='nearest', cmap='RdBu_r', vmin=-0.75, vmax=0.75)
plt.colorbar(label='correlation')
plt.title('Seed-point correlation (centre neuron)')
plt.tight_layout()
plt.savefig('example_correlation.pdf')
plt.show()
