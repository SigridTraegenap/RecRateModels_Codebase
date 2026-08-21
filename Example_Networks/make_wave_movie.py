#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Movies from a moving-stimulus run (run_wave_input_batched.py).
==============================================================

Same style as plot_timestability_v2.py, with one addition that matters for a
time-varying input: a side-by-side STIMULUS | ACTIVITY movie, so you can see the
network response against the bar that drove it. The two panels stay in register
because the recording grid is aligned to the stimulus grid -- activity record r
belongs to stimulus frame r // records_per_frame.

    cd Example_Networks
    python make_wave_movie.py                        # defaults: group 4, trial 0, stim 0
    python make_wave_movie.py --group 0 --stim 3 --trial 2
    python make_wave_movie.py --list                 # what is in the file
    python make_wave_movie.py --no-filter --formats mp4,gif
    python make_wave_movie.py --zoom 10 50 10 50     # crop, as in the original script

Outputs (into the current folder):
    wave_stim_activity_g{group}_t{trial}_s{stim}.mp4 / .gif   side-by-side
    wave_activity_g..._t..._s....gif                          activity + contour
    wave_contour_overlay_g..._t..._s....pdf                    all contours, one figure
    wave_contour_buildup_g..._t..._s....gif                    contours accumulating

NOTE ON MATPLOTLIB >= 3.10
--------------------------
plot_timestability_v2.py uses `cs.collections`, which was deprecated in 3.8 and
REMOVED in 3.10 (this machine has 3.10.9), so that script's ContourBuildup
section would now raise AttributeError. Since 3.8 a ContourSet is itself an
Artist, so `cs.set_visible(False)` replaces the loop over `cs.collections`.
The same applies to `cm.get_cmap(name, N)`, removed in 3.9 -- use
`matplotlib.colormaps[name].resampled(N)`.
"""

import sys
import argparse
from os.path import abspath, sep, pardir

path_parent = abspath('') + sep + pardir + sep
sys.path.append(path_parent)

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tools_2D.simulate.global_params import global_path
from tools_general.filter_funcs import lowhigh_filter_stack

parser = argparse.ArgumentParser()
parser.add_argument('--save-key', default='wave_batched_demo')
parser.add_argument('--group', default='4', help='HDF5 group index')
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--stim', type=int, default=0, help='bar direction index')
parser.add_argument('--list', action='store_true',
                    help='list the groups in the file and exit')
parser.add_argument('--no-filter', action='store_true',
                    help='skip the bandpass (lowhigh_filter_stack) applied by default')
parser.add_argument('--sig-high', type=float, default=6)
parser.add_argument('--sig-low', type=float, default=1)
parser.add_argument('--fps', type=float, default=10)
parser.add_argument('--formats', default='mp4,gif')
parser.add_argument('--percentile', type=float, default=70,
                    help='contour level, as a percentile of each frame')
parser.add_argument('--zoom', type=int, nargs=4, default=None,
                    metavar=('X0', 'X1', 'Y0', 'Y1'))
parser.add_argument('--vmin-zero', action='store_true',
                    help='force vmin=0 instead of the data minimum '
                         '(the original script notes this changes the look)')
args = parser.parse_args()

act_file = global_path + 'activity_v{}.hdf5'.format(args.save_key)
par_file = global_path + 'NetworkParams_v{}.hdf5'.format(args.save_key)

# ---------------------------------------------------------------- listing
if args.list:
    with h5py.File(act_file, 'r') as f:
        print("groups in {}:".format(act_file))
        for g in sorted(f.keys(), key=lambda s: int(s) if s.isdigit() else s):
            if 'activity' in f[g]:
                sh = f[g]['activity'].shape
                print("  group {:<3} activity {}  "
                      "-> --trial 0..{}  --stim 0..{}".format(
                          g, sh, sh[0] - 1, sh[1] - 1))
            else:
                print("  group {:<3} (empty -- a run that failed before saving)".format(g))
    sys.exit(0)

# ---------------------------------------------------------------- load
with h5py.File(act_file, 'r') as f:
    if args.group not in f or 'activity' not in f[args.group]:
        sys.exit("group {!r} has no activity. Run with --list to see what is "
                 "available.".format(args.group))
    activity = f[args.group]['activity'][:]        # (n_trial, n_stim, n_tp, n_pop, N, M)

with h5py.File(par_file, 'r') as f:
    grp = f[args.group]
    inputs_flat = grp['inputs'][:]                 # (n_sims, n_frames * N * M)
    records_per_frame = int(grp['records_per_frame'][()]) \
        if 'records_per_frame' in grp else 1
    record_every = int(grp['record_every'][()]) if 'record_every' in grp else 1

n_trial, n_stim, n_tp, n_pop, N, M = activity.shape
if not (0 <= args.trial < n_trial and 0 <= args.stim < n_stim):
    sys.exit("trial/stim out of range: this group has {} trials x {} stimuli"
             .format(n_trial, n_stim))

n_frames_stim = inputs_flat.shape[1] // (N * M)
stim_movie = inputs_flat.reshape(-1, n_frames_stim, N, M)
# flat sim index matches the run script's reshape: trial * n_stim + stim
stim = stim_movie[args.trial * n_stim + args.stim]

act = activity[args.trial, args.stim, :, 0]        # (n_tp, N, M)

print("group {}  trial {}  stim {}".format(args.group, args.trial, args.stim))
print("  activity {}   stimulus {}   {} records per stimulus frame".format(
    act.shape, stim.shape, records_per_frame))

if not args.no_filter:
    act = lowhigh_filter_stack(act.astype(np.float64), mask=None,
                               sig_high=args.sig_high, sig_low=args.sig_low)
    print("  bandpass applied (sig_high={}, sig_low={})".format(
        args.sig_high, args.sig_low))

vmin = 0. if args.vmin_zero else float(np.min(act))
vmax = float(np.max(act))
tag = 'g{}_t{}_s{}'.format(args.group, args.trial, args.stim)
formats = [s.strip() for s in args.formats.split(',') if s.strip()]
interval = 1000.0 / args.fps


def save(ani, stem):
    for ext in formats:
        out = '{}.{}'.format(stem, ext)
        try:
            ani.save(out, fps=args.fps)
            print("  wrote", out)
        except Exception as exc:                      # e.g. ffmpeg missing for mp4
            print("  SKIPPED {} ({}: {})".format(out, type(exc).__name__,
                                                 str(exc)[:80]))


def frame_to_stim(f):
    """Stimulus frame driving activity record f (grids are aligned)."""
    return min(f // records_per_frame, stim.shape[0] - 1)


# ================================================================ 1. side by side
fig, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(7, 3.6))
im_s = ax_s.imshow(stim[0], cmap='binary_r', origin='lower',
                   vmin=float(stim.min()), vmax=float(stim.max()))
im_a = ax_a.imshow(act[0], cmap='binary_r', origin='lower', vmin=vmin, vmax=vmax)
ax_s.set_title('stimulus', fontsize=10)
ax_a.set_title('activity', fontsize=10)
for a in (ax_s, ax_a):
    a.set_xticks([]); a.set_yticks([])
    if args.zoom:
        a.axis(args.zoom)
txt = fig.text(0.5, 0.02, '', ha='center', fontsize=9)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.09, wspace=0.04)


def update_pair(f):
    im_s.set_array(stim[frame_to_stim(f)])
    im_a.set_array(act[f])
    txt.set_text('step {}   (stimulus frame {}/{})'.format(
        (f + 1) * record_every, frame_to_stim(f) + 1, stim.shape[0]))
    return [im_s, im_a, txt]


print("side-by-side stimulus + activity:")
save(FuncAnimation(fig, update_pair, frames=n_tp, interval=interval, blit=False),
     'wave_stim_activity_' + tag)
plt.close(fig)

# ================================================================ 2. activity + contour
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(act[0], cmap='binary_r', origin='lower', vmin=vmin, vmax=vmax)
ax.set_xticks([]); ax.set_yticks([])
ax.contour(act[0] > np.percentile(act[0], args.percentile),
           colors='C1', linewidths=2)          # first frame, as a fixed reference
fig.subplots_adjust(0, 0, 1, 1)
if args.zoom:
    ax.axis(args.zoom)

print("activity with reference contour:")
save(FuncAnimation(fig, lambda f: [im.set_array(act[f])] and [im],
                   frames=n_tp, interval=interval, blit=True),
     'wave_activity_' + tag)
plt.close(fig)

# ================================================================ 3./4. contours
frame_step = max(1, n_tp // 40)
frames_to_plot = np.arange(0, n_tp, frame_step)
# matplotlib >= 3.9: cm.get_cmap(name, N) is gone
cmap = matplotlib.colormaps['plasma'].resampled(len(frames_to_plot))

fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(0, 0, 1, 1)
for i, t in enumerate(frames_to_plot):
    ax.contour(act[t], levels=[np.percentile(act[t], args.percentile)],
               colors=[cmap(i / max(len(frames_to_plot) - 1, 1))],
               linewidths=0.5, alpha=0.7, origin='lower')
ax.set_xticks([]); ax.set_yticks([]); ax.axis('square'); ax.axis('off')
if args.zoom:
    ax.axis(args.zoom)
out = 'wave_contour_overlay_{}.pdf'.format(tag)
fig.savefig(out, dpi=300)
print("  wrote", out)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(0, 0, 1, 1)
contours = []
for i, t in enumerate(frames_to_plot):
    cs = ax.contour(act[t], levels=[np.percentile(act[t], args.percentile)],
                    colors=[cmap(i / max(len(frames_to_plot) - 1, 1))],
                    linewidths=1.5, origin='lower')
    cs.set_visible(False)      # ContourSet is an Artist since mpl 3.8
    contours.append(cs)
ax.set_xticks([]); ax.set_yticks([]); ax.axis('square'); ax.axis('off')
if args.zoom:
    ax.axis(args.zoom)


def update_buildup(k):
    contours[k].set_visible(True)
    return contours[:k + 1]


print("contour buildup:")
save(FuncAnimation(fig, update_buildup, frames=len(frames_to_plot),
                   interval=interval, blit=False),
     'wave_contour_buildup_' + tag)
plt.close(fig)
print("done.")
