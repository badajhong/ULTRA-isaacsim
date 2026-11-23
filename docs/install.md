# Installation

ULTRA simulates with **Isaac Sim 5.1.0** through **Isaac Lab 2.3.2** (Python 3.11, PyTorch 2.7.0 with CUDA 12.8,
`rl-games==1.1.4`). Tested on Ubuntu 24.04 with RTX PRO 6000 Blackwell GPUs (driver 580). Isaac Sim needs an NVIDIA
RTX GPU and a recent driver, see the
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).

## 1. One-command setup

```bash
scripts/setup_isaacsim_env.sh
conda activate ultra-isaacsim
```

The script creates the conda env `ultra-isaacsim` and installs, in this order, PyTorch (cu128), the Isaac Sim pip
package, Isaac Lab (editable, from a git checkout) and `requirements.txt`. It reuses an existing env of the same
name, so it can be re-run. Options are environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENV_NAME` | `ultra-isaacsim` | conda env name |
| `ISAACLAB_PATH` | `../IsaacLab` (next to this repo) | Isaac Lab checkout; cloned at `v2.3.2` if missing |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cu128` | PyTorch wheel index |
| `ACCEPT_EULA` | `0` | `1` accepts the NVIDIA Omniverse EULA for this Isaac Sim install |

The cu128 PyTorch build is required on RTX 50xx / RTX PRO Blackwell GPUs and also works on older GPUs.

## 2. Manual setup

These are the steps the script runs:

```bash
conda create -n ultra-isaacsim python=3.11 -y
conda activate ultra-isaacsim
pip install --upgrade pip

# 1. PyTorch first, so that Isaac Sim does not pull a CUDA build without Blackwell kernels
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# 2. Isaac Sim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# 3. Isaac Lab (core package only)
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
pip install -e ../IsaacLab/source/isaaclab -c scripts/isaacsim_constraints.txt
# 4. ULTRA
pip install -r requirements.txt -c scripts/isaacsim_constraints.txt
```

`scripts/isaacsim_constraints.txt` holds the versions Isaac Sim pins (numpy, torch, click, typing_extensions,
websockets, packaging). Always pass it with `-c` when you install more packages into this env, otherwise pip may
upgrade them.

`environment.yml` describes the same env for `conda env create -f environment.yml`. Isaac Lab still has to be
installed afterwards (step 3).

### NVIDIA Omniverse EULA

The first launch of Isaac Sim asks you to accept the
[NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
Answer the prompt once in an interactive shell, or set `OMNI_KIT_ACCEPT_EULA=YES`. Batch jobs without a terminal
fail with `EOF when reading a line` until the EULA has been accepted.

### Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('Isaac Sim OK'); app.close()"
```

## 3. Isaac Sim notes

* **Backend.** `ultra/env/sim/` builds the scene with Isaac Lab (ground plane, one G1 and one object per env) and
  exposes Isaac Gym-style state tensors (same body / DOF order, xyzw quaternions, env-local positions), so the task
  code, configs, motion data and checkpoints keep their format.
* **Asset cache.** The first run converts `g1_29dof.urdf` and the object meshes to USD under
  `ultra/data/assets/usd/` (git-ignored). Delete that folder to force a reconversion. The first start also fills
  the shader and convex-decomposition caches, so it is slower than later starts.
* **Viewer.** Without `--headless` the Isaac Sim window replaces the Isaac Gym viewer. Close the window to quit; the
  Isaac Gym keyboard shortcuts (`V`, `Esc`) are not available. Debug drawing uses `isaacsim.util.debug_draw`, and
  `--save_images` captures the viewport. Add `--enable_cameras` to capture images in headless mode.
* **Command line.** The old Isaac Gym flags (`--sim_device`, `--pipeline`, `--headless`, ...) still work.
  `--physx`, `--num_threads`, `--subscenes` and `--slices` are accepted and ignored.
* **Multi-GPU.** `scripts/train_student_multigpu.sh` works as before: every `torchrun` rank starts its own Isaac
  Sim instance on GPU `LOCAL_RANK`.
* **Physics settings.** The `sim.physx` section of the env YAMLs keeps Isaac Gym's names and maps to PhysX 5:
  solver type and iteration counts, contact / rest offsets, bounce threshold, max depenetration velocity and
  `max_gpu_contact_pairs` (contact buffer). `default_buffer_size_multiplier` is ignored; Isaac Sim buffer sizes
  can be set directly with `gpu_*` keys (for example `gpu_max_rigid_patch_count: 2097152`) if PhysX reports a
  buffer overflow.
* **Differences from Isaac Gym.** Isaac Sim runs PhysX 5 (Isaac Gym used PhysX 4) and its own V-HACD convex
  decomposition, so object masses differ by about 1 % and the G1 has more collision hulls. Rolling and torsion
  friction of the objects have no PhysX 5 material equivalent and are not applied. The four sensor frames without
  inertia in the URDF get 1 g. Policies trained in Isaac Gym, including the included teacher, run unchanged but
  can behave slightly differently. Check them in Isaac Sim before you depend on them.
* **Isaac Gym version.** The last commit that runs on Isaac Gym Preview 4 (Python 3.8) is `146ddd5`.

## 4. MuJoCo (sim2sim only)

`mujoco` (3.x, installed by `requirements.txt`) is enough for `ultra/sim2sim_*.py`. The merged robot+object scene
is generated at runtime into `ultra/data/assets/merge/` (git-ignored).

## 5. Data and checkpoints

See the *Data* section of the top-level README. Datasets go under `InterAct/`, checkpoints under `checkpoints/`
(both git-ignored). Run all commands from the repository root.
