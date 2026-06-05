# Python Environment Setup

The example scripts require a conda environment with TensorFlow and several
scientific Python packages. Below are instructions for creating it from scratch
or reproducing the tested configuration.

---

## Tested configuration

| Package | Tested version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.18 |
| NumPy | 2.0 |
| SciPy | ≥ 1.10 |
| Matplotlib | ≥ 3.7 |
| h5py | ≥ 3.9 |
| tqdm | ≥ 4.60 |
| scikit-image | ≥ 0.21 |
| scikit-learn | ≥ 1.3 |

---

## Create the environment

```bash
conda create -n tf_pyth python=3.10
conda activate tf_pyth

# Core numerical stack
conda install numpy scipy matplotlib h5py tqdm

# TensorFlow (CPU-only; see GPU note below)
pip install tensorflow==2.18

# Image processing — needed for structured stimuli (bars, bumps, etc.)
conda install scikit-image

# Machine learning utilities — needed for analysis_tools
conda install scikit-learn
```

> **GPU note**: if you have a CUDA-capable GPU, replace the pip line with
> `pip install tensorflow[and-cuda]==2.18` and make sure your CUDA and cuDNN
> versions match TensorFlow's requirements. For CPU-only use (which is fine for
> small networks, N ≤ 60), the plain `tensorflow` package works without any
> CUDA setup.

---

## Which packages are needed for what

| Package | Used by |
|---|---|
| `numpy` | everything |
| `tensorflow` | `brain_network_tf.py` (network integration), `bn_tools_tf.py` |
| `scipy` | `generate_noisy_mh.py` (eigenvalue computation via `scipy.linalg`), `generate_input.py`, `save_activity.py` (`scipy.io`) |
| `matplotlib` | plots in the example scripts |
| `h5py` | HDF5 file I/O (`save_activity.py`, `load_results.py`) |
| `tqdm` | progress bars inside `brain_network_tf.py` and the example |
| `scikit-image` (`skimage`) | `StimTypes.py` — only needed if you generate structured stimuli (bars, bumps, curves) via `StimulusClass` |
| `scikit-learn` (`sklearn`) | `analysis_tools/Alignment_funs.py` — only needed for post-hoc analysis |

For just running the basic example (`loop_reload_Amodes_250925.py`) and loading
results (`load_results.py`), you only need the first six packages.

---

## Running scripts

Always activate the environment and `cd` into the script's own folder first:

```bash
conda activate tf_pyth
cd /path/to/Rec_Model_Standard/Example_Networks
python loop_reload_Amodes_250925.py
```

The scripts add the repo root to `sys.path` using the current working directory,
so running from a different folder will silently break the imports. See
`MODIFICATIONS.md` for IDE alternatives (Spyder, VSCode, etc.).

---

## Verifying the installation

```python
import numpy as np
import tensorflow as tf
import scipy, h5py, tqdm, matplotlib

print("numpy    ", np.__version__)
print("tensorflow", tf.__version__)
print("scipy    ", scipy.__version__)
print("h5py     ", h5py.__version__)
```

TensorFlow will print version info on import — confirm it shows 2.18.x and no
CUDA errors if you are running CPU-only.
