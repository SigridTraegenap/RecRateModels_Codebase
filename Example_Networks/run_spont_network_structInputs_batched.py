#!/usr/bin/env python

'''
Example: simulating a 2D recurrent rate network
================================================
Model equation (continuous time, dimensionless τ=1):

    dA/dt = -A + f( W_rec · A + Input )

  A     : activity vector, shape (N*M,)
  W_rec : recurrent connectivity matrix, shape (N*M, N*M)
  Input : external drive, shape (N*M,) — constant over one simulation
  f     : nonlinearity (here rectification = ReLU)

Two simulation blocks are shown:
  1. SPONTANEOUS — unstructured noise input, 100 patterns
  2. EVOKED       — structured input derived from the spontaneous activity

Results are saved to ../res_tf2_2d/ as HDF5 files.
See load_results.py for how to reload and plot them.
See MODIFICATIONS.md for how to change nonlinearity, connectivity, input modes, etc.

BATCHED VARIANT
---------------
Identical to run_spont_network_structInputs.py except for the import of the
network package (and the Save_key / plot filenames, so it cannot overwrite the
trusted script's results).  `diff` the two files to see the whole change.

This version simulates all stimuli concurrently as one batch, which turns the
per-timestep matrix-VECTOR product into a matrix-MATRIX product.  That is much
faster on CPU and is the only shape in which a GPU (tensorflow-metal) can help
at all -- see PERFORMANCE.md.

Results are not bit-identical to the trusted path; run
check_batched_equivalence.py to see by how much and why.

Run this script from its own folder:
    cd Example_Networks
    conda run -n tf_pyth python run_spont_network_structInputs_batched.py
'''

import sys
import time

# =============================================================================
# PATH SETUP
# =============================================================================
# abspath('') is the current working directory (must be Example_Networks/).
# One pardir step resolves to the repo root so that tools_2D and tools_general
# become importable.
#
# Alternative for IDE users (e.g. Spyder): set the project/working directory to
# the repo root, or see MODIFICATIONS.md for other options.
from os.path import abspath, sep, pardir
path_parent = abspath('') + sep + pardir + sep
sys.path.append(path_parent)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tools_2D.simulate_batched.brain_network_tf as bn
from tools_2D.simulate import generate_noisy_mh
import tools_2D.simulate.connectivity_params as connectivity_params_func
from tools_2D.simulate import save_activity
from tools_2D.simulate import generate_input
from tools_general import filter_funcs

# =============================================================================
# SAVE KEY
# =============================================================================
# Unique string tag for this run.  All output files are named after it:
#   ../res_tf2_2d/activity_v{Save_key}.hdf5
#   ../res_tf2_2d/NetworkParams_v{Save_key}.hdf5
# Change this string to avoid overwriting a previous run.
Save_key = 'loop_reload_25-12-12_N60_linear_batched'
filename = 'activity_v{}'.format(Save_key)

# =============================================================================
# NETWORK PARAMETERS
# =============================================================================

# NETWORK SIZE: N×M neurons arranged on a 2D grid (total N*M units)
N, M = 100,100

# SPECTRAL BOUND γ (called nonlin_fac for historical reasons):
#   γ < 1  →  sub-critical: activity decays to zero
#   γ = 1  →  critical: marginal amplification
#   γ > 1  →  super-critical: spontaneous pattern formation
gamma = 1.05

# NONLINEARITY: 'rectification' = ReLU, f(x) = max(0, x)
# See MODIFICATIONS.md for other options (sigmoid, tanh, fermi, …)
nonlinearity_rule = 'rectification'

# NUMERICAL SCHEME: set below when constructing BrainNetwork
# 'runge_kutta' (4th order) is the default and recommended choice.
# 'forward_euler' is faster but less accurate; 'runge_kutta2' is intermediate.

network_params = {
    'input_shape'       : np.array([N, M]),
    'nonlin_fac'        : gamma,
    'nonlinearity_rule' : nonlinearity_rule,
}

# =============================================================================
# SIMULATION TIME
# =============================================================================

# Total simulated time = Ntime_total * tau (tau = 1, dimensionless)
# At dt=0.15, 660 steps ≈ 1 second of simulated time.
Ntime_total  = 400          # total time in units of tau
dt           = 0.15
runtime_steps = int(Ntime_total / dt)   # total integration steps

# How many timepoints to save from each simulation
# (evenly spaced; here the last and the mid-point)
timepoints = 2

network_params.update({
    'dt'      : dt,
    'runtime' : runtime_steps,
})

# =============================================================================
# SPONTANEOUS INPUT PARAMETERS
# =============================================================================

# Input structure: constant_offset + noise
#   input = input_offset * ones  +  input_noise_level * noise
#
# input_offset      — mean (DC) drive; keeps neurons in a moderate activity range
# input_noise_level — amplitude of the spatial fluctuations around the mean
input_offset      = 1.0
input_noise_level = 0.2

# Spatial structure of the noise:
#   spont_space_btw_peaks_max = None  →  white noise (no spatial structure)
#   set both min and max              →  bandpass-filtered noise (preferred scale range)
spont_space_btw_peaks_min = 0.01
spont_space_btw_peaks_max = None   # None = white noise

n_spont_patterns = 100   # number of independent spontaneous patterns to simulate

n_pop = 1   # number of neural populations (1 = single E population)
            # set to 2 for E/I network — see MODIFICATIONS.md

# =============================================================================
# EVOKED INPUT PARAMETERS
# =============================================================================

n_stimuli  = 10    # number of distinct structured stimuli
n_trial    = 10    # number of noisy repetitions per stimulus

input_end_offset    = 1.0                            # constant drive for evoked (mirrors input_offset)
stimulus_strength   = 0.25 * input_noise_level       # amplitude of the structured component
trial_noise_strength = input_noise_level             # noise added on top of structured input
trial_noise_mode    = 'white'

# =============================================================================
# CONNECTIVITY
# =============================================================================
# Step 1: select a connectivity mode and get the default parameter dict.
#         Each mode string maps to a generating function in connectivity_params.py.
#         See MODIFICATIONS.md for a full list of available modes.
connectivity_mode   = 'NComm24_MH_PlusRF'
connectivity_params = connectivity_params_func.get_paramdict_connectivity(connectivity_mode)

# Step 2: override individual parameters as needed.
#   sigmax      — excitatory spatial scale (pixels)
#   inh_factor  — inhibitory/excitatory scale ratio κ
#   amplitude   — global inhibitory weight (0 = excitatory-only)
#   strength_RF — amplitude of random spatial heterogeneity (0 = homogeneous)
#   normalization — if True, preserves E/I balance per neuron
connectivity_params.update({
    'strength_RF' : 0.4,
    'sigma1_bp'   : 2,
    'sigma2_bp'   : 6,
    'normalization': True,
})

# Step 3: call the generating function to build W_rec.
#   The function builds a Mexican-hat (MH) kernel and multiplies it by a
#   random perturbation field (RF), introducing spatial heterogeneity.
#   Returns w_rec of shape (N*M, N*M).
index = save_activity.gimme_index('{}.hdf5'.format(filename))
print('Index = {}'.format(index))

gen_function = connectivity_params['generating_function']
w_rec, (Patterns) = gen_function(N, M, connectivity_params,
                                 index=index + 1,   # determines random seed
                                 version=Save_key)

# Step 4: normalize W_rec so that max(Re(eigenvalues)) = γ.
#   This is optional — skip if your generating function already sets the
#   desired spectral radius (see MODIFICATIONS.md).
w_rec, kmax, all_eigenvals = generate_noisy_mh.get_kmax_eigvals(w_rec, network_params)

# =============================================================================
# BRAIN NETWORK SETUP
# =============================================================================
# Wrap W_rec and the integration parameters into a BrainNetwork object.
# The constructor compiles a TensorFlow graph for the chosen integrator.
#
#   w_rec             — (N*M, N*M) connectivity matrix
#   nonlinearity_rule — activation function (see MODIFICATIONS.md for options)
#   integrator        — numerical scheme: 'runge_kutta' (4th order, recommended),
#                       'forward_euler', or 'runge_kutta2'
#   delta_t           — integration step size
#   tau               — time constant (dimensionless, = 1)
#   tsteps            — total number of integration steps per simulation
total_t = network_params['runtime']

integrator = 'runge_kutta'
print('Using integrator:', integrator)

bnn = bn.BrainNetwork(
    w_rec             = w_rec,
    nonlinearity_rule = network_params['nonlinearity_rule'],
    integrator        = integrator,
    delta_t           = network_params['dt'],
    tau               = 1.,
    tsteps            = network_params['runtime'],
    data_type         = np.float32,
)

# =============================================================================
# SPONTANEOUS INPUTS
# =============================================================================
# Build an array of inputs with shape (n_spont_patterns, N*M).
# Each row is one independent input pattern:
#   input = input_offset * ones  +  input_noise_level * noise
#
# noise options:
#   white noise (default, spont_space_btw_peaks_max=None):
#       additive_noise = input_noise_level * np.random.randn(n_spont_patterns, N*M)
#   bandpass-filtered noise (set min/max spatial period in pixels):
#       uses filter_funcs.get_additivenoise_ideal(...)

inputs_all = []
for i in range(n_pop):
    if spont_space_btw_peaks_max is not None:
        # Bandpass-filtered noise: only spatial frequencies in [min, max] period
        additive_noise = input_noise_level * filter_funcs.get_additivenoise_ideal(
            (n_spont_patterns, N, M), index=index,
            npatterns=1, seed=81622,
            dist_btw_peaks_max=spont_space_btw_peaks_max,
            dist_btw_peaks_min=spont_space_btw_peaks_min)
        additive_noise = additive_noise.reshape(n_spont_patterns, N * M)
    else:
        # White noise (spatially unstructured)
        additive_noise = input_noise_level * np.random.randn(n_spont_patterns, N * M)

    constant_input = input_offset * np.ones((N * M))
    inputs_pop = (constant_input[None, :] + additive_noise).reshape(n_spont_patterns, N * M)
    inputs_all.append(inputs_pop)

inputs_all = np.asarray(inputs_all)                        # (n_pop, n_spont_patterns, N*M)
inputs_all = np.clip(inputs_all, a_min=0, a_max=np.inf)   # inputs are non-negative

# Reshape to (nsim_total, n_pop*N*M) — the flat format expected by res_Input_mat.
# The extra dimensions (trial, stimulus) are tracked separately in input_shape_woneuron.
inputs_all = np.reshape(inputs_all, (1,) + inputs_all.shape)           # add fake trial dim
inputs_full = np.moveaxis(inputs_all, [0, 1, 2], [0, 2, 1])           # → (trial, stim, pop, N*M)
input_shape_woneuron = inputs_full.shape[:-2]
inputs = inputs_full.reshape(-1, n_pop * N * M)
nsim_total = inputs.shape[0]

plt.imshow(inputs[-1].reshape(N, M), interpolation='nearest', cmap='binary_r')
plt.colorbar()
plt.title('Example spontaneous input pattern')
plt.savefig('test_Input_spont_batched.pdf')
plt.show()

# =============================================================================
# RUN NETWORK — SPONTANEOUS
# =============================================================================
# bnn.res_Input_mat(inputs, total_t, timepoints) runs all n patterns in a loop
# and returns activity of shape (nsim_total, timepoints, n_pop*N*M).
# It always starts from zero initial conditions.
#
# To use random or non-zero initial conditions, replace with the explicit loop:
#
#   activity = np.empty((nsim_total, timepoints, n_pop, N, M)) * np.nan
#   rng = np.random.RandomState(index * 2)
#   for i in tqdm(range(nsim_total)):
#       start_activity = rng.rand(n_pop * N * M) * start_activity_strength
#       iinput = inputs[i].reshape(1, n_pop * N * M)
#       iactivity = bnn.run(iinput, start_activity)         # → tf.Tensor (total_t, n_pop*N*M)
#       iactivity = iactivity.numpy().reshape(total_t, n_pop, N, M)
#       activity[i] = iactivity[total_t // timepoints - 1::total_t // timepoints]

start = time.time()

activity_flat = bnn.res_Input_mat(inputs, total_t, timepoints=timepoints)
# shape: (nsim_total, timepoints, n_pop*N*M)
activity = activity_flat.reshape(nsim_total, timepoints, n_pop, N, M)

plt.imshow(activity[-1, -1, 0], interpolation='nearest', cmap='binary_r')
plt.colorbar()
plt.title('Example spontaneous activity pattern')
plt.savefig('test_spont_pattern_batched.pdf')
plt.show()

# =============================================================================
# SAVE — SPONTANEOUS
# =============================================================================
# save_activity writes two HDF5 files under global_path = ../res_tf2_2d/:
#   activity_v{Save_key}.hdf5       — the activity array
#   NetworkParams_v{Save_key}.hdf5  — all entries of network_params as datasets
#
# Both are organised by folder_index groups (0/, 1/, 2/, …) so multiple runs
# can be accumulated in one file without overwriting.
#
# Saved activity shape: (n_trials, n_stim, n_timepoints, n_pop, N, M)

del connectivity_params['generating_function']  # not serialisable to HDF5
activity = activity.reshape(input_shape_woneuron + (timepoints, n_pop, N, M))
print('Spontaneous simulation done. Shape:', activity.shape)

network_params.update({'inputs': inputs})
network_params = {**network_params, **connectivity_params}

save_activity.save_activity(activity, network_params, w_rec,
                            Save_key, folder_index='{}/'.format(index))

index_load_spont = int(index)   # keep a copy of the spont index for reload below

# =============================================================================
# EVOKED: STRUCTURED INPUTS
# =============================================================================
# Structured stimuli are derived from the spontaneous activity saved above.
# get_endogenous_pattern() loads the HDF5 file, extracts dominant spatial
# modes (here: activity patterns binarised above the 68th percentile),
# and returns n_stimuli patterns of shape (n_stimuli, N*M).
#
# See MODIFICATIONS.md for other endogen_mode options ('purePC', 'CorrPattern',
# 'Activity', 'BP', …) or for using a fully external stimulus array.

endogen_mode = 'Activity'
print('\nBuilding evoked inputs:', endogen_mode)

inputs_all = []
for i in range(n_pop):
    base_stimuli = generate_input.get_endogenous_pattern(
        n_stimuli,
        endogen_mode,
        network_params,
        filename,
        index_load_spont,
    )
    base_stimuli *= stimulus_strength
    constant_input = input_end_offset * np.ones((N * M))
    inputs_pop = (constant_input[None, :] + base_stimuli).reshape(n_stimuli, N * M)
    inputs_all.append(inputs_pop)

inputs_all = np.asarray(inputs_all)                        # (n_pop, n_stim, N*M)

inputs_wonoise = np.copy(inputs_all)

# Add trial-by-trial noise on top of each structured stimulus.
# generate_trials returns shape (n_trials, n_pop, n_stim, N*M).
inputs_all = generate_input.generate_trials(
    inputs_all,
    Ntrial=n_trial,
    noise_strength=trial_noise_strength,
    noise_mode=trial_noise_mode,
)

# Flatten to (nsim_total, n_pop*N*M) for res_Input_mat
inputs_full = np.moveaxis(inputs_all, [0, 1, 2], [0, 2, 1])   # → (trial, stim, pop, N*M)
print('Evoked input shape:', inputs_full.shape)

input_shape_woneuron = inputs_full.shape[:-2]
inputs_full[inputs_full < 0] = 0.
inputs = inputs_full.reshape(-1, n_pop * N * M)
nsim_total = inputs.shape[0]

plt.imshow(inputs_wonoise[0, -1].reshape(N, M), interpolation='nearest', cmap='binary_r')
plt.colorbar()
plt.title('Example evoked input pattern')
plt.savefig('test_InputE_batched.pdf')
plt.show()

# =============================================================================
# RUN NETWORK — EVOKED
# =============================================================================
# Same approach as spontaneous — res_Input_mat handles the loop internally.
# See the comment in the spontaneous block for the equivalent explicit loop.

index += 1   # save under a new group index

activity_flat = bnn.res_Input_mat(inputs, total_t, timepoints=timepoints)
activity = activity_flat.reshape(nsim_total, timepoints, n_pop, N, M)

plt.imshow(activity[-1, -1, 0], interpolation='nearest', cmap='binary_r')
plt.colorbar()
plt.title('Example evoked activity pattern')
plt.savefig('testE_evoked_batched.pdf')
plt.show()

# =============================================================================
# SAVE — EVOKED
# =============================================================================
activity = activity.reshape(input_shape_woneuron + (timepoints, n_pop, N, M))
print('Evoked simulation done. Shape:', activity.shape)

network_params.update({
    'inputs'        : inputs,
    'inputs_wonoise': inputs_wonoise,
    'eigenvals'     : all_eigenvals,
    'kmax'          : kmax,
})

save_activity.save_activity(activity, network_params, w_rec,
                            Save_key, folder_index='{}/'.format(index))

end = time.time()
print('Done. Total time: {:.1f} s'.format(end - start))
print('Results saved to: ../res_tf2_2d/activity_v{}.hdf5'.format(Save_key))
print('See load_results.py to reload and visualise the output.')
