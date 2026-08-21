#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moving stimulus (drifting bar) on the BATCHED path, with time-resolved output.
===========================================================================

A batched counterpart to run_network_waveinput_withMod_v251016.py. Same model,
same stimulus, same recording grid -- but all trials and directions are
simulated concurrently, so it is fast enough to be worth running on the GPU.

    cd Example_Networks
    ~/anaconda3/envs/ampl_tf/bin/python  run_wave_input_batched.py            # CPU
    ~/anaconda3/envs/tf_metal/bin/python run_wave_input_batched.py            # Metal
    ~/anaconda3/envs/tf_metal/bin/python run_wave_input_batched.py --cpu      # CPU inside the metal env
    ~/anaconda3/envs/ampl_tf/bin/python  run_wave_input_batched.py --quick    # tiny smoke test
    ~/anaconda3/envs/tf_metal/bin/python run_wave_input_batched.py --full     # 16 dirs x 5 trials

RUNNING FROM SPYDER: restart the kernel (Ctrl+.) after any edit to
tools_2D/simulate_batched/. Spyder keeps modules in sys.modules between runs and
its User Module Reloader does not reliably reload submodules, which leaves the
package half-updated. The batched package raises a clear ImportError if it
detects this, but a restart avoids it entirely.

Defaults to 4 directions x 2 trials (8 simulations) at N=M=60 -- enough to see the
stimulus sweep and time the path, without the minutes of CPU time a full sweep
costs. --full restores the original wave script's 16 x 5.

WHAT IS DIFFERENT FROM THE ORIGINAL WAVE SCRIPT
-----------------------------------------------
The original builds, per simulation, a full per-timestep input array by
np.repeat-ing each stimulus phase over `step_factor` steps, then calls
bnn.run(iinput_full, ...) once per simulation -- 80 sequential simulations, each
materialising a (1980, 3600) input and a (1980, 3600) trajectory.

Here the stimulus is passed as a *movie*, shape (n_sims, n_frames, n_neurons),
with `steps_per_frame` saying how long each frame is held. No per-step input
array is ever built: 60 frames instead of 1980 rows, a 33x reduction. And the
two grids are decoupled -- the stimulus advances every 33 steps while activity
is recorded every 20 -- which is exactly what the original does via its
`N_itv_save` slice, but without keeping the other 1881 timesteps around.

TIMING is reported separately for setup and simulation. Setup is dominated by a
dense eigendecomposition of the 3600x3600 connectivity matrix (get_kmax_eigvals),
which is pure SciPy on the CPU and identical in both environments -- so compare
the SIMULATION number when judging whether metal helps.
"""

import sys
import time
import argparse
from os.path import abspath, sep, pardir

path_parent = abspath('') + sep + pardir + sep
sys.path.append(path_parent)

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help='hide the GPU from TensorFlow')
parser.add_argument('--quick', action='store_true',
                    help='tiny network too (24x24) -- smoke test only')
parser.add_argument('--n-stimuli', type=int, default=None,
                    help='number of bar directions (default 4)')
parser.add_argument('--n-trials', type=int, default=None,
                    help='trials per direction (default 2)')
parser.add_argument('--full', action='store_true',
                    help='the original wave script\'s 16 directions x 5 trials '
                         '(80 sims). Slow on CPU -- roughly 25 s of simulation '
                         'at N=60, versus ~2.5 s for the default 8.')
parser.add_argument('--no-save', action='store_true', help='skip writing HDF5')
parser.add_argument('--show', action='store_true', help='pop up the figures')
args = parser.parse_args()

import numpy as np
import matplotlib
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
if args.cpu:                      # must happen before any other TF call
    tf.config.set_visible_devices([], 'GPU')

import tools_2D.simulate_batched.brain_network_tf as bn     # <- the batched path
from tools_2D.simulate import generate_noisy_mh
import tools_2D.simulate.connectivity_params as connectivity_params_func
from tools_2D.simulate import save_activity
from tools_2D.simulate import generate_input
from tools_2D.InputClass.Input_class import StimulusClass

# =============================================================================
# PARAMETERS  (mirroring run_network_waveinput_withMod_v251016.py)
# =============================================================================
Save_key = 'wave_batched_demo'
filename = 'activity_v{}'.format(Save_key)

N, M = (24, 24) if args.quick else (60, 60)
gamma = 1.2                  # strength of recurrent interactions
heterogeneity = 0.4          # strength of the perturbation field

# Deliberately small by default: the point of this script is to check that a
# moving stimulus runs (and how fast), not to produce a dataset. The trusted
# path costs ~2.1 s per simulation at N=60, so a full 80-simulation sweep is
# minutes of CPU time. Scale up with --n-stimuli / --n-trials or --full.
if args.full:
    n_stimuli, n_trials = 16, 5           # the original wave script's sweep
else:
    n_stimuli, n_trials = 4, 2
n_stimuli = args.n_stimuli or n_stimuli
n_trials  = args.n_trials or n_trials
n_phasepoints = 12 if args.quick else 60  # stimulus frames per sweep

trial_noise_level = 0.03
input_offset      = 0.
input_scale       = 1

# --- timing -----------------------------------------------------------------
Ntime_total      = 3         # s of simulated time (approximately; see below)
dt               = 0.15
steps_per_second = 132 if args.quick else 660

# Records per stimulus frame.
#
# The recording grid is kept ALIGNED with the stimulus grid -- record_every
# divides steps_per_frame exactly -- so every frame gets the same number of
# samples at the same relative phases, and the last sample of each frame is
# exactly the end-of-frame state. Activity records then map one-to-one onto
# stimulus frames: activity[:, records_per_frame-1::records_per_frame] is the
# end-of-frame state for each of the n_phasepoints frames.
#
# The original wave script does NOT have this property (stimulus every 33 steps,
# N_itv_save=20), so its samples land at drifting phases within each frame. The
# batched path supports that too -- record_every need not divide steps_per_frame
# -- but aligned is easier to interpret.
records_per_frame = 2

# Round steps_per_frame up to a multiple of records_per_frame so the grids align.
# This nudges the total simulated time slightly; the exact value is printed below.
_target_total   = int(steps_per_second * Ntime_total)
_target_spf     = max(records_per_frame, int(round(_target_total / n_phasepoints)))
steps_per_frame = int(np.ceil(_target_spf / records_per_frame) * records_per_frame)
record_every    = steps_per_frame // records_per_frame
runtimesteps_total = steps_per_frame * n_phasepoints
timepoints      = runtimesteps_total // record_every

assert steps_per_frame % records_per_frame == 0
assert runtimesteps_total % record_every == 0
assert timepoints == n_phasepoints * records_per_frame

network_params = {
    'input_shape'       : np.array([N, M]),
    'nonlinearity_rule' : 'rectification',
    'dt'                : dt,
    'runtime'           : runtimesteps_total,
    'nonlin_fac'        : gamma,
}

devices = [d.device_type for d in tf.config.list_logical_devices()]
print()
print("tensorflow {}   devices {}".format(tf.__version__, devices))
print("network    {}x{} = {} neurons".format(N, M, N * M))
print("stimulus   {} directions x {} trials = {} simulations{}".format(
    n_stimuli, n_trials, n_stimuli * n_trials,
    "" if (args.full or args.n_stimuli or args.n_trials)
    else "   (use --full or --n-stimuli/--n-trials for more)"))
print("timing     {} steps = {:.2f} s simulated".format(
    runtimesteps_total, runtimesteps_total / steps_per_second))
print("           stimulus advances every {} steps ({} frames)".format(
    steps_per_frame, n_phasepoints))
print("           activity recorded every {} steps ({} records "
      "= {} per frame, ALIGNED)".format(
          record_every, timepoints, records_per_frame))
print()

t_setup = time.time()

# =============================================================================
# CONNECTIVITY
# =============================================================================
index = save_activity.gimme_index('{}.hdf5'.format(filename))
print('Index = {}'.format(index)); sys.stdout.flush()

connectivity_mode = 'NComm24_MH_PlusRF'
connectivity_params = connectivity_params_func.get_paramdict_connectivity(connectivity_mode)
connectivity_params.update({
    'strength_RF'  : heterogeneity,
    'sigma1_bp'    : 1.5,
    'sigma2_bp'    : 4.5,
    'sigmax'       : 1.54,
    'inh_factor'   : 2,
    'normalization': True,
})
n_pop = 1

gen_function = connectivity_params['generating_function']
w_rec, (Patterns) = gen_function(N, M, connectivity_params,
                                 index=index + 1, version=Save_key)

print("normalising spectral bound to gamma={} "
      "(dense eigendecomposition, the slow part of setup)...".format(gamma))
sys.stdout.flush()
w_rec, kmax, all_eigenvals = generate_noisy_mh.get_kmax_eigvals(w_rec, network_params)

bnn = bn.BrainNetwork(
    w_rec             = w_rec,
    nonlinearity_rule = network_params['nonlinearity_rule'],
    integrator        = 'runge_kutta',
    delta_t           = network_params['dt'],
    tau               = 1.,
    tsteps            = network_params['runtime'],
    data_type         = np.float32,
)

# =============================================================================
# STIMULUS MOVIE
# =============================================================================
numeric_params = {
    'size_visualspace'   : N,
    'size_stimulated_VF' : N,
    'dx_visualfield'     : 1,
}
stim_params = {
    'stim_type'         : 'Bar',
    'visual_angles'     : np.arange(0, 360, 360 / n_stimuli),
    'phases'            : np.linspace(0, 1, n_phasepoints),
    'bar_width'         : max(4, N // 3),
    'bar_length'        : None,
    'periodic_stimulus' : True,
}

StimulusGenerator = StimulusClass(numeric_params)
inputs_full = StimulusGenerator.generate_stimuli(stim_params)   # (n_stim, n_phase, N, M)
inputs_full = inputs_full * input_scale + input_offset

# NOTE: generate_trials draws its noise from np.random.default_rng() with no
# seed, so the trial noise differs on every run and cannot be fixed with
# np.random.seed(). See KNOWN_ISSUES.md section 1.
Inputs_full_trials = generate_input.generate_trials(
    inputs_full, Ntrial=n_trials, noise_strength=trial_noise_level)
Inputs_full_trials[Inputs_full_trials < 0] = 0.

# (n_trials, n_stim, n_phase, N, M) -> (n_sims, n_frames, n_neurons)
# which is exactly the layout res_Input_mat_movie expects.
movie = Inputs_full_trials.reshape(n_trials * n_stimuli, n_phasepoints, N * M)
nsim_total = movie.shape[0]
input_shape_woneuron = (n_trials, n_stimuli)

setup_elapsed = time.time() - t_setup
print("setup done in {:.1f} s".format(setup_elapsed))
print("stimulus movie {}  ({:.1f} MB) -- the per-step equivalent would be "
      "{:.1f} MB".format(movie.shape, movie.nbytes / 1e6,
                         movie.nbytes / 1e6 * steps_per_frame))
print()

# =============================================================================
# SIMULATE  — all directions and trials at once, time-resolved
# =============================================================================
t_sim = time.time()
activity_flat = bnn.res_Input_mat_movie(
    movie,
    steps_per_frame = steps_per_frame,
    record_every    = record_every,
    n_steps         = runtimesteps_total,
)                                    # (n_sims, timepoints, n_neurons)
sim_elapsed = time.time() - t_sim

print()
print("=" * 66)
print("SIMULATION  {:.2f} s total   ->  {:.4f} s per simulation".format(
    sim_elapsed, sim_elapsed / nsim_total))
print("            ({} sims x {} steps, {} neurons, on {})".format(
    nsim_total, runtimesteps_total, N * M,
    'GPU' if 'GPU' in devices else 'CPU'))
print("setup       {:.1f} s (SciPy eigendecomposition, CPU either way)".format(
    setup_elapsed))
if nsim_total < 32:
    print("-" * 66)
    print("NOTE: {} simulations is below the batching knee (~32-128), so the".format(
        nsim_total))
    print("      per-simulation cost above is NOT representative -- W is being")
    print("      re-read from memory for too few simulations at a time. Expect")
    print("      roughly 5x better per-simulation with --full. See PERFORMANCE.md,")
    print("      'How big should batch_size be?'.")
print("=" * 66)
print()

activity = activity_flat.reshape(nsim_total, timepoints, n_pop, N, M)
print('Activity shape:', activity.shape,
      '({:.1f} MB)'.format(activity.nbytes / 1e6))
print('finite:', np.all(np.isfinite(activity)),
      '  range [{:.3f}, {:.3f}]'.format(activity.min(), activity.max()))

# Because the grids are aligned, end-of-frame states line up with the stimulus.
end_of_frame = activity[:, records_per_frame - 1::records_per_frame]
assert end_of_frame.shape[1] == n_phasepoints, (end_of_frame.shape, n_phasepoints)
print('end-of-frame states:', end_of_frame.shape,
      '-> one per stimulus frame, directly comparable to the movie')

# =============================================================================
# PLOTS
# =============================================================================
n_show = min(4, n_stimuli)
n_cols = min(10, timepoints)
col_idx = np.linspace(0, timepoints - 1, n_cols).astype(int)

fig, axs = plt.subplots(nrows=n_show, ncols=n_cols,
                        figsize=(1.2 * n_cols, 1.3 * n_show), squeeze=False)
for istim in range(n_show):
    for j, itp in enumerate(col_idx):
        axs[istim, j].imshow(activity[istim, itp, 0], interpolation='nearest')
        axs[istim, j].axis('off')
        if istim == 0:
            axs[istim, j].set_title('step {}'.format((itp + 1) * record_every),
                                    fontsize=6)
fig.suptitle('Activity: {} directions (rows) over time (columns)'.format(n_show))
plt.savefig('wave_batched_activity.pdf', bbox_inches='tight')
if args.show:
    plt.show()

movie_plot = movie.reshape(n_trials, n_stimuli, n_phasepoints, N, M)
fcols = np.linspace(0, n_phasepoints - 1, n_cols).astype(int)
fig, axs = plt.subplots(nrows=n_show, ncols=n_cols,
                        figsize=(1.2 * n_cols, 1.3 * n_show), squeeze=False)
for istim in range(n_show):
    for j, f in enumerate(fcols):
        axs[istim, j].imshow(movie_plot[0, istim, f], interpolation='nearest')
        axs[istim, j].axis('off')
fig.suptitle('Stimulus movie: {} directions (rows) over frames (columns)'.format(n_show))
plt.savefig('wave_batched_input.pdf', bbox_inches='tight')
if args.show:
    plt.show()

# =============================================================================
# SAVE
# =============================================================================
if not args.no_save:
    del connectivity_params['generating_function']
    activity_save = activity.reshape(
        input_shape_woneuron + (timepoints, n_pop, N, M))
    network_params.update({'inputs': movie.reshape(nsim_total, -1),
                           'eigenvals': all_eigenvals,
                           'kmax': kmax,
                           'steps_per_frame': steps_per_frame,
                           'record_every': record_every,
                           'records_per_frame': records_per_frame})
    network_params = {**network_params, **connectivity_params}
    save_activity.save_activity(activity_save, network_params, w_rec, Save_key,
                                folder_index='{}/'.format(index))
    print('Saved to ../res_tf2_2d/activity_v{}.hdf5 (group {})'.format(Save_key, index))

print('Figures: wave_batched_activity.pdf, wave_batched_input.pdf')
