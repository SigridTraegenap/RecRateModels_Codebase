# 2D Network StructConn — System Overview

Collaborative modelling project (Sigrid Traegenap et al.).
Models a 2D cortical sheet with structured recurrent connectivity driven by
visual-like inputs. The main focus is how lateral connectivity shape determines
spontaneous and evoked activity statistics.

---

## Environment & running rules

- **Conda environment: `tf_pyth`** (TensorFlow 2.18, NumPy 2.0).
- **Always `cd` into the script's own folder before running** — `global_params.py`
  and `sys.path` additions use `abspath('')` (current working directory), so
  running a script from a different folder silently breaks path resolution.

```bash
cd Example_Networks
conda run -n tf_pyth python loop_reload_Amodes_250925.py
```

### Two simulation packages

`tools_2D/simulate` (trusted, `tf.scan`) and `tools_2D/simulate_batched`
(batched, `tf.while_loop`) expose the same class name, constructor signature, and
`res_Input_mat()` signature and return shape, so you switch with one import line:

```python
import tools_2D.simulate.brain_network_tf         as bn   # trusted
import tools_2D.simulate_batched.brain_network_tf as bn   # batched
```

The batched package simulates all stimuli at once, which is **7.5x faster on CPU**
and the only form in which an Apple GPU helps at all — under `tensorflow-metal`
the trusted path is ~100x *slower* than CPU, while the batched path is the
fastest option available. It supports `integrator='runge_kutta'` only and never
materialises the full trajectory.

**See [PERFORMANCE.md](PERFORMANCE.md)** for the measured numbers, why the
original shape defeats a GPU, and how far the two paths' results diverge.

---

## Repository layout

```
Example_Networks/
  run_spont_network_structInputs.py ← annotated example: connectivity + spont + evoked simulation
  run_spont_network_structInputs_batched.py ← same, on the batched (fast) path
  check_batched_equivalence.py      ← verifies the batched path against the trusted one
  bench_paths.py                    ← times both paths (add --cpu to hide the GPU)
  load_results.py                   ← minimal script for loading and plotting saved HDF5 output
  MODIFICATIONS.md                  ← guide for changing nonlinearity, connectivity, inputs, etc.
  ENVIRONMENT_SETUP.md              ← conda environment setup and package requirements

tools_2D/
  simulate/                 ← TRUSTED path: one stimulus at a time, tf.scan
    brain_network_tf.py     ← BrainNetwork class (TF integrator)
    integration_methods_tf.py ← forward_euler, runge_kutta, runge_kutta2
    bn_tools_tf.py          ← nonlinearity functions (rectification, sigmoid, …)
    generate_noisy_mh.py    ← connectivity generators + get_kmax_eigvals
    connectivity_params.py  ← get_paramdict_connectivity(conn_mode)
    save_activity.py        ← HDF5 save functions + gimme_index
    generate_input.py       ← generate_trials (add trial noise)
    global_params.py        ← global_path for HDF5 output
    random_Gaussfields.py   ← bandpass random fields for RF perturbation
  InputClass/
    Input_class.py          ← StimulusClass.generate_stimuli(stim_params)
    StimTypes.py            ← per-stimulus generators (Bar, MeanderCurve, Bumps, …)
    ReceptiveField_funcs.py ← helper geometry (get_center, getEllipsoid)

analysis_tools/             ← helper functions for post-hoc analysis

tools_general/              ← shared utilities (filter_funcs, etc.)

res_tf2_2d/                 ← HDF5 output files (NOT in git; created at runtime)
```

---

## Network dynamics

Continuous-time rate model integrated numerically:

```
τ · dA/dt = -A + f( W_rec · A + Input )
```

- **τ = 1** (dimensionless), **dt = 0.15**, integrator = Runge-Kutta 4th order.
- **f = rectification** (ReLU): `f(x) = max(0, x)`.
- State `A` is a flat vector of length `N*M` (row-major reshape of the 2D grid).
- `Input` has shape `(ksteps, N*M)` — one input frame per timestep.
- `steps_per_second = 660` (660 steps ≈ 1 s of simulated time).

### BrainNetwork constructor (brain_network_tf.py)

```python
bnn = BrainNetwork(
    w_rec          = w_rec,           # (N*M, N*M) float32
    nonlinearity_rule = 'rectification',
    integrator     = 'runge_kutta',
    delta_t        = 0.15,
    tau            = 1.0,
    tsteps         = runtimesteps_total,  # int
    data_type      = np.float32,
)
activity = bnn.run(iinput_full, start_activity)
# returns tf.Tensor shape (tsteps, N*M)
```

`PlasticBrainNetwork` extends `BrainNetwork` with BCM-style feedforward weight
learning (not used in the stats sweep).

---

## Connectivity pipeline

### Step 1 — generate W_rec

All connectivity modes ultimately return `(w_rec, meta)` where
`w_rec` has shape `(N*M, N*M)`.

The **main function** used in the stats sweep is
`generate_noisy_mh.homogenMH_plusRF_wrap`:

1. Build a **homogeneous Mexican-hat** kernel:
   `W_MH[i,j] = G(Δ, σ=sigmax) − amplitude · G(Δ, σ=sigmax·inh_factor)`
   with periodic boundary conditions.
   - `sigmax` — excitatory spatial scale (pixels)
   - `inh_factor` — ratio of inhibitory to excitatory scale (κ in manuscript)
   - `amplitude` — global inhibitory weight (set 0 for excitatory-only)
   - `inh_factor=np.inf` → no inhibitory term (Stage 1)

2. Multiply by a **random perturbation field** (RF):
   `W = W_MH × (1 + strength_RF · φ)`
   where `φ` is a bandpass Gaussian random field with scales
   `[sigma1_bp, sigma2_bp]`.
   - `perturb_samepat=True` → same field applied to every row (NatComm paper)
   - `strength_RF=0` → homogeneous MH (no heterogeneity)

3. **Optional normalization** (`normalization=True`):
   Scale excitatory weights of each neuron so exc/inh balance is preserved.
   Falls back to std-normalisation when `normalization=False` and
   `normalization_fallback=True`.

### Step 2 — eigenvalue normalisation (spectral bound γ)

```python
w_rec, kmax, all_eigenvals = generate_noisy_mh.get_kmax_eigvals(w_rec, network_params)
```

Scales `w_rec` so that `max(Re(eigenvalues)) = nonlin_fac` (= γ).

- **γ < 1**: sub-critical, activity decays.
- **γ = 1**: critical, marginal amplification.
- **γ > 1**: super-critical, spontaneous activity / pattern formation.
- `nonlin_fac` is set in `network_params['nonlin_fac']` and passed via
  `sweep_utils.run_and_save(..., nonlin_fac=...)`.
- **Minimum N ≈ 15–20** for MH normalisation to work — the MH kernel needs
  enough pixels to have a non-zero dominant eigenvalue.

### Available connectivity modes (connectivity_params.py)

| `conn_mode` | Description |
|---|---|
| `Stage1_ExcOnly` | Exc-only MH + RF (sigmax=1, amplitude=0, inh_factor=inf) |
| `NComm24_MH_PlusRF` | Full MH + RF (NatComm 2024 params, sigmax=1.8→1.53 in sweep) |
| `homogenMH` | Homogeneous MH, no RF (sigmax=1.8, inh_factor=2.5) |
| `NN_18_elongMH` | Heterogeneous elongated MH (ecc=0.8) |
| `Mod_mult_dym` | Modular connectivity (random covariance structure) |
| `EI_MH` | Explicit E/I neuron populations |

All modes support keyword overrides after `get_paramdict_connectivity`:
```python
params = connectivity_params_func.get_paramdict_connectivity(conn_mode)
params.update(conn_overrides)   # override individual fields
```

---

## Input generation

### StimulusClass (tools_2D/InputClass/Input_class.py)

```python
StimulusGenerator = StimulusClass({'size_visualspace': N,
                                    'size_stimulated_VF': N,
                                    'dx_visualfield': 1})
inputs = StimulusGenerator.generate_stimuli(stim_params)
# shape: (n_stimuli, n_phasepoints, N, M)  — clipped to ≥0
```

`stim_params` dict must contain `'stim_type'` plus type-specific keys:

| `stim_type` | Key params |
|---|---|
| `Fullfield` | none — returns ones, angle-independent |
| `Bar` | `bar_width`, `bar_length`, `visual_angles`, `phases`, `periodic_stimulus` |
| `BumpyBar` | as Bar + `bump_amplitude`, `bump_frequency` |
| `Ellipse` | `bar_width`, `bar_length`, `visual_angles`, `phases` |
| `MeanderCurve` | `bar_width`, `curvedness`, `meander_amp`, `meander_freq`, `visual_angles`, `phases` |
| `Bumps` | `spacing` (pixel gap), `n_rows`, `sigma`, `visual_angles`, `phases` |

### Coordinate conventions (important!)

`get_center(Size, pos, angle)` returns `(cx, cy)` where:
- `cx` → **row** index
- `cy` → **column** index

`getEllipsoid(Size, Pos=(p0, p1))` treats:
- `p0` → **column**, `p1` → **row**

Bar long-axis direction in (row, col) space = `(sin(angle_rad), cos(angle_rad))`.

---

## Saving (save_activity.py / global_params.py)

```python
save_activity.save_activity(activity, save_params, w_rec, Save_key,
                             folder_index='0/')
```

Writes to `global_path + 'activity_v{Save_key}.hdf5'`.

`global_path` resolves to `<cwd>/../res_tf2_2d/` — this is why scripts must be
run from their own subdirectory.

HDF5 structure:
```
activity_v{key}.hdf5
  0/
    activity  — shape (n_trials, n_stim, n_timepoints, 1, N, M) float32
    shape     — same as above
NetworkParams_v{key}.hdf5
  0/
    nonlin_fac, dt, runtime, inputs, eigenvals, kmax, pertubation_field, …
```

`gimme_index(filename)` auto-increments the folder index (0, 1, 2, …) for
multi-run accumulation in one file. The stats sweep bypasses this by using
`folder_index='0/'` and unique `Save_key` per condition.

---

## Analysis conventions

- **Activity colormap**: `binary_r` — white = high, black = 0 (minimum).
- **Correlation colormap**: `RdBu_r`, symmetric ±0.75.
- **Seed-point correlation**:
  ```python
  flat = activity[:, :, cutoff:, 0].reshape(-1, N*M)
  corr = np.corrcoef(flat.T).reshape(N, M, N, M)
  corr_map = corr[N//2, M//2]   # (N, M) for centre seed pixel
  ```
  Discard first `cutoff=3` timepoints (transient).
- `N_itv_save = 20` — activity is stored every 20 integration steps, giving
  `timepoints = runtimesteps_total // 20` saved frames.

---

## Key gotchas

| Problem | Cause | Fix |
|---|---|---|
| `global_path` wrong directory | Script run from wrong folder | `cd` into script folder first |
| HDF5 "name already exists" | Re-running with same Save_key | Delete old file or use unique run_id |
| Eigenvalue normalisation fails (NaN) | N too small for MH kernel | Use N ≥ 20 |
| Input length mismatch | Non-integer step_factor | Use `max(1, round(...))` + trim/pad |
| `tools_2D` not found | sys.path uses `abspath('')` | Run from script's own directory |
