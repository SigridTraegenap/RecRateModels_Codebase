# Modifications Guide

Common extensions beyond the basic example in `loop_reload_Amodes_250925.py`.

---

## Changing the nonlinearity

In the example, change the `nonlinearity_rule` key in `network_params` and the
corresponding argument in `BrainNetwork(...)`:

```python
network_params = { ..., 'nonlinearity_rule': 'rectification', ... }

bnn = bn.BrainNetwork(w_rec=w_rec,
                      nonlinearity_rule='rectification',  # ← change here
                      ...)
```

Available options (defined in `tools_2D/simulate/bn_tools_tf.py`):

| `nonlinearity_rule` | Description |
|---|---|
| `'rectification'` | ReLU: `f(x) = max(0, x)` — threshold-linear, most commonly used |
| `'linear'` | Identity: `f(x) = x` — no saturation, useful for linear analysis |
| `'sigmoid'` | Standard logistic: `f(x) = 1/(1+e^{-x})` — bounded output in [0,1] |
| `'sigmoid_v2'` | Shifted sigmoid: `f(x) = 2·max(0, sigmoid(x)−0.5)` — zero below threshold |
| `'tanh'` | Hyperbolic tangent: output in [−1,1] |
| `'fermi'` | Very steep sigmoid (~step): `f(x) = sigmoid(50x)` |
| `'clipping'` | `f(x) = clip(x, 0, 1)` — bounded rectification |
| `'rectification_squared'` | `f(x) = max(0, x)²` — supralinear gain |
| `'rectification_threshold'` | `f(x) = max(0, x−0.1)` — rectification with threshold offset |
| `'shallow_rectification'` | `f(x) = max(0, 0.1·x)` — very shallow gain |

---

## Changing the connectivity scheme

The connectivity mode string selects a generating function and a default parameter
dict from `tools_2D/simulate/connectivity_params.py`. To switch:

```python
connectivity_mode = 'NComm24_MH_PlusRF'   # ← change to any mode below
connectivity_params = connectivity_params_func.get_paramdict_connectivity(connectivity_mode)
connectivity_params.update({'sigmax': 2.0})  # override individual params
```

Available modes:

| `conn_mode` | Description |
|---|---|
| `'NComm24_MH_PlusRF'` | Mexican-hat + random perturbation field (NatComm 2024 params) — **default in example** |
| `'Stage1_ExcOnly'` | Excitatory-only MH + RF (`amplitude=0`, no inhibition) |
| `'homogenMH'` | Homogeneous Mexican-hat, no RF (purely translation-invariant) |
| `'NN_18_elongMH'` | Heterogeneous MH with elongated (orientation-selective) kernels (ecc=0.8) |
| `'EI_MH'` | Explicit E and I neuron populations with separate MH kernels |
| `'Mod_mult_dym'` | Modular connectivity with random covariance structure |

Key parameters to tune for MH-type connectivities:

| Parameter | Effect |
|---|---|
| `sigmax` | Excitatory spatial scale (pixels); larger → broader excitation |
| `inh_factor` | Ratio of inhibitory to excitatory scale (κ); larger → wider inhibition |
| `amplitude` | Global inhibitory weight; `0` = excitatory-only |
| `strength_RF` | Amplitude of random heterogeneity; `0` = perfectly homogeneous |
| `sigma1_bp`, `sigma2_bp` | Spatial frequency range of the RF perturbation field |
| `normalization` | If `True`, preserves E/I balance per neuron after RF multiplication |

### Adding a new connectivity class

1. Write a generating function with this signature:
   ```python
   def my_connectivity(N, M, connectivity_params, index=0, version='', **kwargs):
       # build w_rec with shape (N*M, N*M)
       return w_rec, meta   # meta can be None or any auxiliary output
   ```

2. Register it in `connectivity_params.py` by adding a new `if conn_mode == '...':`
   block with `'generating_function': my_connectivity`.

3. In the example script, set `connectivity_mode = 'my_new_mode'`.

---

## Skipping eigenvalue normalization

The call to `generate_noisy_mh.get_kmax_eigvals(w_rec, network_params)` rescales
`w_rec` so that the largest real part of any eigenvalue equals γ (`nonlin_fac`).
This step is **optional** — skip it if:
- your generating function already returns a matrix with the desired spectral radius, or
- you prefer to control the weight scale manually (e.g. by simply multiplying `w_rec`
  by a scalar).

To skip, remove or comment out:
```python
w_rec, kmax, all_eigenvals = generate_noisy_mh.get_kmax_eigvals(w_rec, network_params)
```
and pass the unscaled `w_rec` directly to `BrainNetwork`.

Note: normalization requires N ≥ 20 to work reliably (the MH kernel needs enough
pixels for a non-zero dominant eigenvalue).

---

## Two-population (E/I) networks

Use `connectivity_mode = 'EI_MH'` which generates a block-structured weight matrix
for explicit excitatory and inhibitory populations.

In the example, set `n_pop = 2`. The input array then has shape
`(n_pop, n_patterns, N*M)` and the second population's drive can be scaled
independently via `EI_input_modulation`:
```python
n_pop = 2
EI_input_modulation = 0.8   # relative drive to the inhibitory population
if n_pop == 2:
    inputs_all[1] *= EI_input_modulation
```

---

## Different structured input modes

`generate_input.get_endogenous_pattern()` derives structured stimuli from
previously saved spontaneous activity. The `endogen_mode` argument selects
the extraction method:

| `endogen_mode` | Description |
|---|---|
| `'Activity'` | Random spont frames, binarised above the 68th percentile — **default in example** |
| `'purePC'` | Leading singular vectors of the activity covariance matrix |
| `'purePC_firstN'` | First N singular vectors (contiguous, not sampled) |
| `'CorrPattern'` | Seed-point correlation maps from random seed locations |
| `'random'` | Phase-shuffled spontaneous frames (same spectrum, scrambled spatial content) |
| `'BP'` | Bandpass-filtered white noise (uses `spont_space_btw_peaks_min/max`) |

To use a fully external stimulus instead, skip `get_endogenous_pattern()` entirely
and supply any array of shape `(n_stimuli, N*M)` as `base_stimuli`:
```python
base_stimuli = your_stimulus_array   # shape (n_stimuli, N*M), values ~O(1)
```

---

## Python path setup for IDEs

The script contains:
```python
path_parent = abspath('') + sep + pardir + sep + pardir + sep
sys.path.append(path_parent)
```

This resolves to the **repo root** (two levels up from `Example_Networks/`) and
must be run **from the `Example_Networks/` folder** to work correctly.

Alternatives:
- **Spyder**: set *Tools → Preferences → Python interpreter → Working directory*
  (or the console's working directory) to the repo root, then the path block
  resolves to itself and the imports still work.
- **Any IDE / terminal**: add the repo root to `PYTHONPATH` before running:
  ```bash
  export PYTHONPATH=/path/to/Rec_Model_Standard:$PYTHONPATH
  python Example_Networks/loop_reload_Amodes_250925.py
  ```
- **Inside a script**: replace the os.path block with a direct `sys.path` insert:
  ```python
  sys.path.insert(0, '/path/to/Rec_Model_Standard')
  ```
