#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batched BrainNetwork -- drop-in replacement for
tools_2D/simulate/brain_network_tf.py

Switch between the two paths with a single import line:

    import tools_2D.simulate.brain_network_tf         as bn   # trusted, tf.scan
    import tools_2D.simulate_batched.brain_network_tf as bn   # batched, tf.while_loop

The constructor signature and `res_Input_mat()` (signature AND return shape) are
identical to the original, so nothing else in a calling script needs to change.

WHY
---
The original simulates one stimulus at a time. Each RK4 stage is then a rank-1
matrix-vector product against a 51.8 MB W (at N=M=60), i.e. ~0.5 FLOP per byte
loaded -- pure memory bandwidth. On Apple Silicon the CPU and GPU share one
memory bus, so a GPU has no headroom to win there, only per-dispatch overhead to
lose. Batching the (mutually independent) stimuli turns the GEMV into a GEMM and
raises arithmetic intensity by roughly the batch size, which is the regime where
tensorflow-metal actually pays off. It is also substantially faster on CPU.

DIFFERENCES FROM THE TRUSTED PATH
---------------------------------
* integrator='runge_kutta' only (RK4). The others raise a clear error.
* run() is not available -- the full trajectory is never materialised. Use
  res_Input_mat() (same API as the original) or the advance() primitive.
* Results are NOT bit-identical to the original: batched matmul and the
  original's tensordot+transpose use different kernels and reduction orders, so
  float32 results differ in the last bits. See
  Example_Networks/check_batched_equivalence.py.
"""

from os.path import abspath, sep, pardir
import sys
sys.path.append(abspath('') + sep + pardir + sep + pardir + sep)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Reuse the nonlinearities from the original package rather than duplicating
# them -- they are shared, and a divergence between the two copies would be a
# silent source of mismatch between the trusted and batched paths.
from tools_2D.simulate import bn_tools_tf as bnt
from . import integration_methods_tf as im

# Fail fast and legibly on a half-reloaded package.
#
# Spyder and IPython keep modules in sys.modules between runs, and Spyder's User
# Module Reloader does not reliably reload submodules -- so this file can be the
# new version while integration_methods_tf is still the copy imported before an
# edit. Without this check the symptom is an AttributeError raised from inside
# autograph-generated code, which points nowhere useful.
_REQUIRED = ('runge_kutta_explicit',
             'runge_kutta_explicit_trajectory',
             'runge_kutta_explicit_movie',
             'runge_kutta_explicit_movie_strided')
_missing = [_f for _f in _REQUIRED if not hasattr(im, _f)]
if _missing:
    raise ImportError(
        "tools_2D.simulate_batched.integration_methods_tf is missing: {}\n"
        "\n"
        "This almost always means a STALE module is cached in a long-running "
        "interpreter -- the source on disk is fine.\n"
        "\n"
        "  Fix: restart the kernel.\n"
        "       Spyder:  Consoles -> Restart kernel   (Ctrl+.)\n"
        "       IPython: exit and start a new session\n"
        "\n"
        "Module was loaded from:\n  {}".format(", ".join(_missing), im.__file__))

print("numpy -V", np.__version__)
print("tf -V", tf.__version__)


_NONLINEARITIES = {
    'linear':                bnt.nl_linear_tf,
    'rectification':         bnt.nl_rect_tf,
    'rectification_threshold': bnt.nl_rect_th_tf,
    'shallow_rectification': bnt.nl_shallow_rect_tf,
    'clipping':              bnt.nl_clip_tf,
    'sigmoid':               bnt.nl_sigmoid_tf,
    'sigmoid_v2':            bnt.nl_sigmoidcustom_tf,
    'tanh':                  bnt.nl_tanh_tf,
    'fermi':                 bnt.nl_fermi_tf,
    'rectification_squared': bnt.nl_rect_squared_tf,
}


class BrainNetwork:
    def __init__(self,
                 w_rec,
                 nonlinearity_rule,
                 integrator='runge_kutta',
                 delta_t=0.01,
                 tau=10.,
                 tsteps=None,
                 data_type=np.float32):

        if isinstance(data_type, str):
            data_type = np.dtype(data_type).type
        self.data_type_np = data_type

        if integrator != 'runge_kutta':
            raise NotImplementedError(
                "tools_2D.simulate_batched supports integrator='runge_kutta' "
                "only, got {!r}. Use tools_2D.simulate for the "
                "others.".format(integrator))
        self.integrator = integrator

        if w_rec is None:
            raise ValueError("w_rec is required")
        self.w_rec = np.asarray(w_rec).astype(self.data_type_np)
        self.num_neurons = self.w_rec.shape[0]

        self.delta_t = self.data_type_np(delta_t)
        self.tsteps = None if tsteps is None else np.int32(tsteps)
        if isinstance(tau, np.ndarray):
            self.tau = tau.astype(self.data_type_np)
        else:
            self.tau = self.data_type_np(tau)

        self.nonlinearity_rule = nonlinearity_rule
        self._init_nonlinearity()

        self.tf_w_rec = tf.constant(self.w_rec, name="w_rec")
        self.tf_delta_t = tf.constant(self.delta_t, name="deltat")
        self.tf_tau = tf.constant(self.tau, name="tau")

        self._init_tf_computations()

    def _init_nonlinearity(self):
        try:
            self.tf_nonlinearity = _NONLINEARITIES[self.nonlinearity_rule]
        except KeyError:
            raise Exception('Unknown nonlinearity rule: {!r}. Options: {}'.format(
                self.nonlinearity_rule, sorted(_NONLINEARITIES)))

    def _check_variables(self):
        if (self.tf_w_rec is None or
                self.nonlinearity_rule is None or
                self.integrator is None or
                self.tf_delta_t is None or
                self.tf_tau is None):
            raise Exception("Not all required settings set.")

    def _init_tf_computations(self):
        # n_steps is a plain Python int, so tf.function retraces once per
        # distinct value -- which is exactly what we want: it bakes a static
        # trip count and static shapes into each concrete graph.
        #
        # Deliberately NOT using reduce_retracing=True. That relaxes shapes to
        # dynamic, which is the very condition suspected of triggering the
        # tensorflow-metal slowdown in the original path (unknown shapes ->
        # ops falling back to CPU placement -> a device<->host round trip on
        # every timestep). A handful of extra traces costs a few seconds once;
        # dynamic shapes can cost 40x forever. Prefer uniform batch shapes.
        @tf.function
        def advance(inputs, start_activity, n_steps):
            return im.runge_kutta_explicit(
                inputs,
                self.tf_w_rec,
                start_activity,
                self.tf_delta_t,
                self.tf_tau,
                self.tf_nonlinearity,
                n_steps,
            )

        self._advance = advance

        @tf.function
        def advance_trajectory(inputs, start_activity, n_steps, record_every):
            return im.runge_kutta_explicit_trajectory(
                inputs,
                self.tf_w_rec,
                start_activity,
                self.tf_delta_t,
                self.tf_tau,
                self.tf_nonlinearity,
                n_steps,
                record_every,
            )

        self._advance_trajectory = advance_trajectory

        @tf.function
        def advance_movie(inputs_movie, start_activity, steps_per_frame):
            return im.runge_kutta_explicit_movie(
                inputs_movie,
                self.tf_w_rec,
                start_activity,
                self.tf_delta_t,
                self.tf_tau,
                self.tf_nonlinearity,
                steps_per_frame,
            )

        self._advance_movie = advance_movie

        @tf.function
        def advance_movie_strided(inputs_movie, start_activity, steps_per_frame,
                                  record_every, n_steps):
            return im.runge_kutta_explicit_movie_strided(
                inputs_movie,
                self.tf_w_rec,
                start_activity,
                self.tf_delta_t,
                self.tf_tau,
                self.tf_nonlinearity,
                steps_per_frame,
                record_every,
                n_steps,
            )

        self._advance_movie_strided = advance_movie_strided

    def advance(self, inputs, start_activity, n_steps):
        """Integrate a batch forward by n_steps and return the final state.

        inputs, start_activity : array-like (n_sims, num_neurons)
        n_steps : Python int
        Returns a tf.Tensor of shape (n_sims, num_neurons).
        """
        self._check_variables()
        inputs = tf.cast(inputs, dtype=self.data_type_np)
        start_activity = tf.cast(start_activity, dtype=self.data_type_np)
        return self._advance(inputs, start_activity, int(n_steps))

    def run(self, inputs, start_activity):
        raise NotImplementedError(
            "The batched path never materialises the full trajectory: batched, "
            "that is n_sims x tsteps x num_neurons floats (3.8 GB at 100 sims, "
            "2666 steps, 3600 neurons) of which the callers here keep 2 rows.\n"
            "Use res_Input_mat(...) -- same signature and return shape as the "
            "trusted path -- or advance(inputs, start_activity, n_steps) to "
            "step a batch forward and get the final state.\n"
            "If you genuinely need every timestep, use "
            "tools_2D.simulate.brain_network_tf instead.")

    def res_Input_mat(self, inputs, total_t, timepoints=2, start=None,
                      batch_size=None):
        """Simulate every row of `inputs` and return sampled activity.

        Signature and return shape match
        tools_2D.simulate.brain_network_tf.BrainNetwork.res_Input_mat.

        inputs : (nsim_total, num_neurons)
        returns: (nsim_total, timepoints, num_neurons), dtype float64

        batch_size : int or None
            Simulations run concurrently. None (default) runs all at once --
            the batched state is small (100 x 3600 x 4 B ~ 1.4 MB); it is W that
            dominates memory, and W is shared across the batch. Set this only
            if you hit a memory limit. Note that a batch_size which does not
            divide nsim_total leaves a ragged final chunk, costing one extra
            tf.function trace.
        """
        inputs = np.asarray(inputs)
        nsim_total, nneurons = inputs.shape
        if nneurons != self.num_neurons:
            raise ValueError(
                "inputs has {} neurons, network has {}".format(
                    nneurons, self.num_neurons))

        # Reproduce the original's sampling EXACTLY. There, tf.scan output row j
        # is the state after j+1 steps, and the caller slices
        #     iactivity[total_t//timepoints - 1 :: total_t//timepoints]
        # so the recorded states are those after chunk, 2*chunk, ... steps.
        # We integrate in deltas between consecutive sample points instead.
        chunk = total_t // timepoints
        if chunk < 1:
            raise ValueError(
                "total_t ({}) must be >= timepoints ({})".format(total_t, timepoints))
        steps_after = np.arange(chunk - 1, total_t, chunk) + 1
        if len(steps_after) != timepoints:
            raise ValueError(
                "total_t={} and timepoints={} do not produce {} samples: the "
                "original's slice [{}::{}] yields {}. The trusted path fails on "
                "this combination too (it broadcasts into a fixed-width array), "
                "so pick a total_t that is a multiple of timepoints.".format(
                    total_t, timepoints, timepoints, chunk - 1, chunk,
                    len(steps_after)))
        deltas = np.diff(np.concatenate(([0], steps_after)))

        # Cast the whole input array once, not one row per simulation.
        inputs_f = inputs.astype(self.data_type_np)

        if start is not None:
            start = np.asarray(start).astype(self.data_type_np)
            if start.ndim == 1:
                start = np.broadcast_to(start, (nsim_total, nneurons))
            elif start.shape != (nsim_total, nneurons):
                raise ValueError(
                    "start must have shape ({0},) or ({1}, {0}), got {2}".format(
                        nneurons, nsim_total, start.shape))

        # float64 container, matching the original's np.empty(...) default, so
        # downstream HDF5 dtypes are unchanged. np.full rather than the
        # original's np.empty(...)*np.nan, which warns ("invalid value
        # encountered in multiply") whenever the uninitialised memory happens to
        # contain an inf. Same resulting array.
        activity = np.full((nsim_total, timepoints, nneurons), np.nan)

        bs = nsim_total if batch_size is None else int(batch_size)
        print("run!")
        sys.stdout.flush()
        for b0 in tqdm(range(0, nsim_total, bs)):
            b1 = min(b0 + bs, nsim_total)
            batch_inputs = tf.constant(inputs_f[b0:b1])
            if start is None:
                x = tf.zeros((b1 - b0, nneurons), dtype=self.data_type_np)
            else:
                x = tf.constant(np.ascontiguousarray(start[b0:b1]))

            for t_idx, delta in enumerate(deltas):
                x = self._advance(batch_inputs, x, int(delta))
                activity[b0:b1, t_idx, :] = x.numpy()

        return activity

    def res_Input_mat_trajectory(self, inputs, total_t, record_every=1,
                                 start=None, batch_size=None,
                                 out_dtype=np.float32, memory_budget_gb=2.0,
                                 verbose=True):
        """Time-resolved activity: record the state every `record_every` steps.

        Use this when you want the time course, not just a couple of sampled
        timepoints. For only a handful of timepoints, res_Input_mat() is
        cheaper -- it never materialises a trajectory at all.

        inputs : (nsim_total, num_neurons)
        record_every : int
            1 records every integration step (the full trajectory, as the
            trusted tf.scan path returns). total_t must be divisible by it.
        out_dtype : numpy dtype for the returned array
            Defaults to float32. A full trajectory is large and the simulation
            itself runs in float32, so float64 here would double the memory for
            no extra information. (res_Input_mat returns float64 to match the
            trusted path's container; there the array is small.)
        batch_size : int or None
            None auto-picks the largest batch whose on-device trajectory fits
            in memory_budget_gb, so a full-resolution run does not try to
            allocate tens of GB at once.

        returns: (nsim_total, n_records, num_neurons) with
                 n_records = total_t // record_every.
                 Element r is the state after (r+1)*record_every steps.
        """
        inputs = np.asarray(inputs)
        nsim_total, nneurons = inputs.shape
        if nneurons != self.num_neurons:
            raise ValueError("inputs has {} neurons, network has {}".format(
                nneurons, self.num_neurons))
        record_every = int(record_every)
        if record_every < 1:
            raise ValueError("record_every must be >= 1")
        if total_t % record_every != 0:
            raise ValueError(
                "total_t={} is not divisible by record_every={} ({} steps would "
                "be left over and silently dropped). Nearby divisors of "
                "total_t: {}.".format(
                    total_t, record_every, total_t % record_every,
                    sorted(d for d in range(1, total_t + 1)
                           if total_t % d == 0
                           and 0.4 * record_every <= d <= 2.5 * record_every)
                    or "none close; consider adjusting total_t"))
        n_records = total_t // record_every

        # Auto-chunk the batch so the on-device trajectory stays bounded.
        bytes_per_sim = n_records * nneurons * 4
        if batch_size is None:
            budget = int(memory_budget_gb * 1e9)
            bs = max(1, min(nsim_total, budget // max(bytes_per_sim, 1)))
        else:
            bs = int(batch_size)

        out_gb = nsim_total * n_records * nneurons * np.dtype(out_dtype).itemsize / 1e9
        if verbose:
            print("trajectory: {} records x {} sims x {} neurons "
                  "-> {:.2f} GB returned ({}), batch_size={} "
                  "({:.2f} GB on device per batch)".format(
                      n_records, nsim_total, nneurons, out_gb,
                      np.dtype(out_dtype).name, bs, bs * bytes_per_sim / 1e9))
            sys.stdout.flush()

        inputs_f = inputs.astype(self.data_type_np)
        if start is not None:
            start = np.asarray(start).astype(self.data_type_np)
            if start.ndim == 1:
                start = np.broadcast_to(start, (nsim_total, nneurons))
            elif start.shape != (nsim_total, nneurons):
                raise ValueError(
                    "start must have shape ({0},) or ({1}, {0}), got {2}".format(
                        nneurons, nsim_total, start.shape))

        activity = np.empty((nsim_total, n_records, nneurons), dtype=out_dtype)

        for b0 in tqdm(range(0, nsim_total, bs)):
            b1 = min(b0 + bs, nsim_total)
            batch_inputs = tf.constant(inputs_f[b0:b1])
            if start is None:
                x0 = tf.zeros((b1 - b0, nneurons), dtype=self.data_type_np)
            else:
                x0 = tf.constant(np.ascontiguousarray(start[b0:b1]))
            # (n_records, batch, neurons) -> (batch, n_records, neurons)
            traj = self._advance_trajectory(batch_inputs, x0, int(total_t),
                                            record_every)
            activity[b0:b1] = np.transpose(traj.numpy(), (1, 0, 2)).astype(out_dtype)

        return activity

    def res_Input_mat_movie(self, inputs_movie, steps_per_frame, start=None,
                            batch_size=None, out_dtype=np.float32,
                            memory_budget_gb=2.0, verbose=True,
                            record_every=None, n_steps=None):
        """Simulate with a TIME-VARYING input, e.g. a moving stimulus.

        Frame f of the movie drives the network for `steps_per_frame`
        integration steps, then frame f+1 takes over; the state at the end of
        each frame is returned. Total simulated steps = n_frames *
        steps_per_frame.

        This matches the trusted path's `runge_kutta` semantics exactly: there,
        passing `run()` an (n_steps, n_neurons) array makes tf.scan feed one row
        per step, and all four RK stages of a step see the same row. Here that
        is the steps_per_frame=1 case.

        (Be aware the trusted `forward_euler` does NOT honour a time-varying
        input -- it silently uses only the first row, because of a typo in
        tools_2D/simulate/integration_methods_tf.py:43. Only `runge_kutta`
        works there, and only `runge_kutta` exists here.)

        inputs_movie : (n_sims, n_frames, num_neurons)
            Per-simulation stimulus movie. To drive every simulation with the
            same movie, np.broadcast_to it to this shape first.
        steps_per_frame : int
            Integration steps each frame is held for. Because the four RK
            stages within a step share one input, keep this small enough that
            the stimulus barely moves within a frame.
        out_dtype : numpy dtype of the returned array (default float32).

        record_every : int or None
            None (default) records at the end of each frame, giving one record
            per frame. Give an integer to record on an independent grid -- e.g.
            a stimulus advancing every 33 steps while activity is written out
            every 20. Must divide n_steps.
        n_steps : int or None
            Total integration steps. Defaults to n_frames * steps_per_frame.
            Only meaningful together with record_every; if larger than the movie
            covers, the last frame is held for the remainder.

        returns: (n_sims, n_records, num_neurons).
                 With record_every=None, n_records == n_frames and element f is
                 the state at the end of frame f. Otherwise n_records ==
                 n_steps // record_every and element r is the state after
                 (r+1) * record_every steps. Take [:, -1, :] for the final state.
        """
        inputs_movie = np.asarray(inputs_movie)
        if inputs_movie.ndim != 3:
            raise ValueError(
                "inputs_movie must be (n_sims, n_frames, num_neurons), got shape "
                "{}. For a constant input use res_Input_mat() instead.".format(
                    inputs_movie.shape))
        nsim_total, n_frames, nneurons = inputs_movie.shape
        if nneurons != self.num_neurons:
            raise ValueError("inputs_movie has {} neurons, network has {}".format(
                nneurons, self.num_neurons))
        steps_per_frame = int(steps_per_frame)
        if steps_per_frame < 1:
            raise ValueError("steps_per_frame must be >= 1")

        strided = record_every is not None
        if strided:
            record_every = int(record_every)
            if record_every < 1:
                raise ValueError("record_every must be >= 1")
            n_steps = int(n_steps) if n_steps is not None \
                else n_frames * steps_per_frame
            if n_steps % record_every != 0:
                raise ValueError(
                    "n_steps={} is not divisible by record_every={} ({} steps "
                    "would be left over). Nearby divisors of n_steps: {}.".format(
                        n_steps, record_every, n_steps % record_every,
                        sorted(d for d in range(1, n_steps + 1)
                               if n_steps % d == 0
                               and 0.4 * record_every <= d <= 2.5 * record_every)
                        or "none close"))
            n_records = n_steps // record_every
        elif n_steps is not None:
            raise ValueError("n_steps is only meaningful together with record_every")
        else:
            n_steps = n_frames * steps_per_frame
            n_records = n_frames

        # Device-side cost per simulation: the movie in, the trajectory out.
        bytes_per_sim = (n_frames + n_records) * nneurons * 4
        if batch_size is None:
            budget = int(memory_budget_gb * 1e9)
            bs = max(1, min(nsim_total, budget // max(bytes_per_sim, 1)))
        else:
            bs = int(batch_size)

        if verbose:
            print("movie: {} sims x {} frames x {} steps/frame = {} steps total; "
                  "{} records (every {} steps); batch_size={} "
                  "({:.2f} GB on device per batch)".format(
                      nsim_total, n_frames, steps_per_frame, n_steps,
                      n_records, record_every if strided else steps_per_frame,
                      bs, bs * bytes_per_sim / 1e9))
            sys.stdout.flush()

        movie_f = inputs_movie.astype(self.data_type_np)
        if start is not None:
            start = np.asarray(start).astype(self.data_type_np)
            if start.ndim == 1:
                start = np.broadcast_to(start, (nsim_total, nneurons))
            elif start.shape != (nsim_total, nneurons):
                raise ValueError(
                    "start must have shape ({0},) or ({1}, {0}), got {2}".format(
                        nneurons, nsim_total, start.shape))

        activity = np.empty((nsim_total, n_records, nneurons), dtype=out_dtype)

        for b0 in tqdm(range(0, nsim_total, bs)):
            b1 = min(b0 + bs, nsim_total)
            # (batch, frames, neurons) -> frame-major (frames, batch, neurons)
            batch_movie = tf.constant(
                np.ascontiguousarray(np.transpose(movie_f[b0:b1], (1, 0, 2))))
            if start is None:
                x0 = tf.zeros((b1 - b0, nneurons), dtype=self.data_type_np)
            else:
                x0 = tf.constant(np.ascontiguousarray(start[b0:b1]))
            if strided:
                traj = self._advance_movie_strided(
                    batch_movie, x0, steps_per_frame, record_every, n_steps)
            else:
                traj = self._advance_movie(batch_movie, x0, steps_per_frame)
            activity[b0:b1] = np.transpose(traj.numpy(), (1, 0, 2)).astype(out_dtype)

        return activity
