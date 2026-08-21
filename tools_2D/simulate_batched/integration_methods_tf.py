#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batched integration methods -- drop-in replacement for
tools_2D/simulate/integration_methods_tf.py

DIFFERENCE TO THE ORIGINAL
--------------------------
The original integrates ONE simulation at a time: the state is a rank-1 vector
of shape (num_neurons,), so `x @ W.T` is a matrix-VECTOR product. Every one of
the ~13M weights in W is read to perform a single multiply-add, i.e. ~0.5 FLOP
per byte loaded. That is entirely memory-bandwidth bound, which is the one
regime where a GPU cannot beat a CPU on Apple Silicon (they share the same
memory bus).

Here the state is (n_sims, num_neurons), so the same op becomes a matrix-MATRIX
product: W is read once per timestep to serve ALL simulations instead of once
per simulation. Arithmetic intensity rises by roughly the batch size, which is
what makes GPU execution (tensorflow-metal) worthwhile.

The original also uses tf.scan, which stacks the state at EVERY timestep. Only
a couple of timepoints are ever kept, so at batch size 100 that would allocate
100 x 2666 x 3600 x 4 bytes = 3.8 GB of trajectory to then discard >99% of it.
tf.while_loop carries only the current state and returns the final one; the
caller re-enters the loop for each timepoint it wants to record.

Only 'runge_kutta' (RK4) is implemented -- see the stubs at the bottom.
"""

import numpy as np
import tensorflow as tf

tf.autograph.set_verbosity(0, True)


def runge_kutta_explicit(inputs,
                         w_rec,
                         start_activity,
                         delta_t,
                         tau,
                         nonlinearity,
                         n_steps):
    """Batched RK4, 4th order. Returns ONLY the final state.

    Parameters
    ----------
    inputs : tensor, shape (n_sims, num_neurons)
        External drive, constant over the integration window.
    w_rec : tensor, shape (num_neurons, num_neurons)
        Recurrent connectivity.
    start_activity : tensor, shape (n_sims, num_neurons)
        Initial condition.
    delta_t, tau : scalar tensors
    nonlinearity : callable f(x) -> x
    n_steps : PYTHON int
        Number of integration steps.

        This MUST be a Python int, never a tensor. A Python int gives the
        while_loop a statically-known trip count and keeps every tensor shape
        in the loop body static. The original passes a tf.constant here, which
        makes the tiled input's shape (None, num_neurons) and leaves the scan
        with a dynamic trip count and a dynamically-sized TensorArray. Under a
        PluggableDevice such as tensorflow-metal, unknown shapes are a common
        trigger for ops falling back to CPU placement -- and a CPU-placed op
        inside the loop body forces a device<->host round trip on EVERY
        timestep. Keeping shapes static avoids that whole class of problem.

    Returns
    -------
    tensor, shape (n_sims, num_neurons) -- the state after n_steps steps.
    """
    def fprime(x):
        # tf.matmul(..., transpose_b=True) instead of the original's
        # tensordot(x, tf.transpose(w_rec)): no explicit transpose op, and a
        # real GEMM once x is 2D.
        return (-x + nonlinearity(inputs + tf.matmul(x, w_rec, transpose_b=True))) / tau

    def body(i, x):
        k1 = fprime(x)
        k2 = fprime(x + 0.5 * k1 * delta_t)
        k3 = fprime(x + 0.5 * k2 * delta_t)
        k4 = fprime(x + k3 * delta_t)
        # NOTE: these coefficients are copied verbatim from the original
        # implementation, including the 1/6 on k4 (textbook RK4 uses 1/3 on k3
        # and k4 and they sum to 1; this scheme's weights sum to 1 as well but
        # distribute differently). The goal here is to reproduce the trusted
        # path's dynamics exactly, NOT to "fix" the integrator. Do not change.
        x_new = x + delta_t * ((1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4)
        return i + 1, x_new

    _, x_final = tf.while_loop(
        cond=lambda i, x: i < n_steps,
        body=body,
        loop_vars=(tf.constant(0), start_activity),
        maximum_iterations=n_steps,
    )
    return x_final


def runge_kutta_explicit_trajectory(inputs,
                                    w_rec,
                                    start_activity,
                                    delta_t,
                                    tau,
                                    nonlinearity,
                                    n_steps,
                                    record_every):
    """Batched RK4 that RECORDS the state every `record_every` steps.

    Same scheme as runge_kutta_explicit, but returns a trajectory instead of
    only the final state.

    Structure is a nested loop: an outer loop over recording points, and an
    inner loop of `record_every` integration steps. That deliberately avoids a
    tf.cond inside the hot loop -- branching every timestep to decide whether to
    record would cost more than the recording does. Everything stays on device
    in one call; nothing round-trips to host until the caller asks.

    Parameters
    ----------
    n_steps, record_every : PYTHON ints (see runge_kutta_explicit on why).
        n_steps must be divisible by record_every; the caller enforces this.

    Returns
    -------
    tensor, shape (n_steps // record_every, n_sims, num_neurons)
        Element r is the state after (r+1) * record_every steps, so the last
        element is the state after n_steps -- matching the sampling convention
        of the trusted path's tf.scan output.
    """
    n_records = n_steps // record_every

    def fprime(x):
        return (-x + nonlinearity(inputs + tf.matmul(x, w_rec, transpose_b=True))) / tau

    def rk4_step(i, x):
        k1 = fprime(x)
        k2 = fprime(x + 0.5 * k1 * delta_t)
        k3 = fprime(x + 0.5 * k2 * delta_t)
        k4 = fprime(x + k3 * delta_t)
        # coefficients copied verbatim from the trusted implementation
        x_new = x + delta_t * ((1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4)
        return i + 1, x_new

    def outer_body(r, x, traj):
        _, x = tf.while_loop(
            cond=lambda i, _x: i < record_every,
            body=rk4_step,
            loop_vars=(tf.constant(0), x),
            maximum_iterations=record_every,
        )
        return r + 1, x, traj.write(r, x)

    trajectory = tf.TensorArray(
        dtype=start_activity.dtype,
        size=n_records,
        dynamic_size=False,
        clear_after_read=False,
        element_shape=start_activity.shape,
    )

    _, _, trajectory = tf.while_loop(
        cond=lambda r, x, traj: r < n_records,
        body=outer_body,
        loop_vars=(tf.constant(0), start_activity, trajectory),
        maximum_iterations=n_records,
    )
    return trajectory.stack()


def runge_kutta_explicit_movie(inputs_movie,
                               w_rec,
                               start_activity,
                               delta_t,
                               tau,
                               nonlinearity,
                               steps_per_frame):
    """Batched RK4 driven by a TIME-VARYING input (e.g. a moving stimulus).

    The input is piecewise constant: frame f drives the network for
    `steps_per_frame` integration steps, then frame f+1 takes over. Set
    steps_per_frame=1 for a new input at every single step.

    Piecewise frames rather than a value per step is the memory-sensible
    choice: a stimulus moves on its own timescale, far slower than delta_t, so
    holding each frame for many steps costs nothing in fidelity while a
    per-step input array of (n_steps, n_sims, n_neurons) would be gigabytes.

    Within one integration step all four RK stages use the same input, which is
    exactly what the trusted path does (its tf.scan hands the same element to
    k1-k4). Note this makes the scheme first-order accurate in the input's time
    variation even though it is 4th order in the dynamics -- keep
    steps_per_frame small enough that the stimulus moves slowly across a frame.

    Parameters
    ----------
    inputs_movie : tensor, shape (n_frames, n_sims, num_neurons)
        Frame-major, so a frame can be gathered cheaply inside the loop.
    steps_per_frame : PYTHON int (see runge_kutta_explicit on why).

    Returns
    -------
    tensor, shape (n_frames, n_sims, num_neurons)
        Element f is the state at the END of frame f, i.e. after
        (f+1) * steps_per_frame steps.
    """
    n_frames = inputs_movie.shape[0]

    def fprime(x, inp):
        return (-x + nonlinearity(inp + tf.matmul(x, w_rec, transpose_b=True))) / tau

    def rk4_step(i, x, inp):
        k1 = fprime(x, inp)
        k2 = fprime(x + 0.5 * k1 * delta_t, inp)
        k3 = fprime(x + 0.5 * k2 * delta_t, inp)
        k4 = fprime(x + k3 * delta_t, inp)
        # coefficients copied verbatim from the trusted implementation
        x_new = x + delta_t * ((1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4)
        return i + 1, x_new, inp

    def outer_body(f, x, traj):
        inp = tf.gather(inputs_movie, f)
        _, x, _ = tf.while_loop(
            cond=lambda i, _x, _inp: i < steps_per_frame,
            body=rk4_step,
            loop_vars=(tf.constant(0), x, inp),
            maximum_iterations=steps_per_frame,
        )
        return f + 1, x, traj.write(f, x)

    trajectory = tf.TensorArray(
        dtype=start_activity.dtype,
        size=n_frames,
        dynamic_size=False,
        clear_after_read=False,
        element_shape=start_activity.shape,
    )

    _, _, trajectory = tf.while_loop(
        cond=lambda f, x, traj: f < n_frames,
        body=outer_body,
        loop_vars=(tf.constant(0), start_activity, trajectory),
        maximum_iterations=n_frames,
    )
    return trajectory.stack()


def _not_implemented(name):
    def _stub(*args, **kwargs):
        raise NotImplementedError(
            "'{0}' has no batched implementation. The batched package "
            "(tools_2D.simulate_batched) currently supports integrator="
            "'runge_kutta' only.\n"
            "For '{0}', use the original package instead:\n"
            "    import tools_2D.simulate.brain_network_tf as bn".format(name)
        )
    return _stub


forward_euler = _not_implemented('forward_euler')
runge_kutta2 = _not_implemented('runge_kutta2')
