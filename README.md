<p align="center">
  <h1 align="center"><strong>ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation</strong></h1>
  <p align="center">
    <a href="https://xialin-he.github.io/" target="_blank">Xialin He</a><sup>*</sup>&emsp;
    <a href="https://sirui-xu.github.io/" target="_blank">Sirui Xu</a><sup>*</sup>&emsp;
    <a href="https://lixinyao11.github.io/" target="_blank">Xinyao Li</a>&emsp;
    <a href="https://runpeidong.web.illinois.edu/" target="_blank">Runpei Dong</a>&emsp;
    <a href="https://scholar.google.com/citations?user=J2z-0lgAAAAJ&hl=en" target="_blank">Liuyu Bian</a>&emsp;
    <a href="https://yxw.cs.illinois.edu/" target="_blank">Yu-Xiong Wang</a><sup>&dagger;</sup>&emsp;
    <a href="https://lgui.cs.illinois.edu/" target="_blank">Liang-Yan Gui</a><sup>&dagger;</sup>
    <br>
    University of Illinois Urbana-Champaign
    <br>
    <sup>*</sup>Equal contribution&emsp;<sup>&dagger;</sup>Equal advising
    <br>
    <strong>IROS 2026 &middot; Best Paper Award on Mobile Manipulation &middot; Finalist 🏆</strong>
  </p>
</p>

<p align="center">
  <a href='https://arxiv.org/abs/2603.03279'>
    <img src='https://img.shields.io/badge/Arxiv-2603.03279-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2603.03279'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://ultrahumanoid.github.io/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://github.com/Sirui-Xu/ULTRA'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>

## 🏠 Overview

<div align="center">
  <img src="assets/ultra_demo.gif" width="100%" alt="ULTRA demo"/>
</div>

> **ULTRA** is a single multimodal controller for humanoid whole-body loco-manipulation: it tracks a motion reference when one is available, and acts from egocentric perception and a sparse goal when it is not — one policy, one set of weights, on a real Unitree G1.

## ⚙️ Installation

ULTRA runs on **Isaac Sim 5.1** through **Isaac Lab 2.3.2** (Python 3.11, PyTorch 2.7 + CUDA 12.8), with `mujoco`
for sim2sim. One script creates the conda environment:

```bash
scripts/setup_isaacsim_env.sh      # conda env `ultra-isaacsim` (ACCEPT_EULA=1 accepts the NVIDIA Omniverse EULA)
conda activate ultra-isaacsim
```

See [docs/install.md](docs/install.md) for the manual steps, the EULA, and the differences from the original Isaac Gym
setup. Run every command below from the repository root. The first run converts the robot and object assets to USD
(cached in `ultra/data/assets/usd/`).

## 📦 Data

| Download | Used for | Extract `.pt` files to |
| --- | --- | --- |
| [InterMimic prepared OMOMO references](https://drive.google.com/file/d/1l2E5qR97Ap8jrLrJPHmtNT8DDW1qKhY_/view?usp=sharing) | SMPL-X retargeting input · `[T, 591]` | `InterAct/OMOMO_retarget/` |
| [ULTRA retargeted and augmented data](https://drive.google.com/file/d/1g6OzxXGJczZ4klVqKCpTzy-orKkllCr4/view?usp=sharing) | G1 teacher/student training · `[T, 630]` | `InterAct/OMOMO_retarget_aug/` |

The released G1 archive contains **OMOMO physics-based retargeted and augmented motions**. The source archive follows [InterMimic's data format](https://github.com/Sirui-Xu/InterMimic#data).

<details>
<summary>Data layout and supported objects</summary>

Place the `.pt` files directly in the directories above. ULTRA includes assets for `largebox`, `plasticbox`, `smallbox`, and `suitcase`; the retargeting script selects motions for these objects.

</details>

## 🚀 Reproduce the pipeline

| Stage | Input → output | Entry point |
| --- | --- | --- |
| 1 · Retarget | Prepared OMOMO SMPL-X → G1 policy | `scripts/train_retarget_smplx.sh` |
| 2 · Augment | SMPL-X + trained policy → G1 rollouts | `scripts/export_retarget_smplx.py` |
| 3 · Train teacher | G1 rollouts → tracking policy | `scripts/train_teacher.sh` |
| 4 · Distill student | G1 rollouts + included teacher → student | `scripts/train_student.sh` |
| 5 · Finetune student | Distilled student → goal-directed policy | `scripts/train_finetune.sh` |

<details>
<summary><strong>1 · Train the retargeting policy</strong></summary>

After extracting the InterMimic `OMOMO_retarget` archive, run from the repository root:

```bash
scripts/train_retarget_smplx.sh
```

The script prepares supported object clips in `InterAct/OMOMO_retarget_supported_080_080_080/` and trains `UltraG1` with **position target PD control**. This data generation stage has no domain randomization, random pushes, or observation noise.

The configuration runs up to 50,000 epochs and writes checkpoints to `output/retarget_smplx/g1_retarget_smplx/nn/`.

</details>

<details>
<summary><strong>2 · Export retargeted and augmented G1 motion</strong></summary>

Use a trained retargeting checkpoint to replay the prepared references. `--xyz X Y Z` scales the sparse trajectories along each axis; repeat it for augmentation variants. `--asset-scale` selects an available object size (`080_080_080` or `100_100_100`).

```bash
python scripts/export_retarget_smplx.py \
  --input-dir InterAct/OMOMO_retarget \
  --checkpoint output/retarget_smplx/g1_retarget_smplx/nn/g1_retarget_smplx.pth \
  --output-dir output/retarget_export_080 \
  --asset-scale 080_080_080 \
  --xyz 1 1 1 --xyz 1.05 1 0.95

# Repeat with the larger released object assets.
python scripts/export_retarget_smplx.py \
  --input-dir InterAct/OMOMO_retarget \
  --checkpoint output/retarget_smplx/g1_retarget_smplx/nn/g1_retarget_smplx.pth \
  --output-dir output/retarget_export_100 \
  --asset-scale 100_100_100 \
  --xyz 1 1 1 --xyz 1.05 1 0.95
```

The exporter writes the resulting G1 motions to the chosen output directory. The released `InterAct/OMOMO_retarget_aug/` archive is ready to use for teacher training.

</details>

<details>
<summary><strong>3–4 · Train a teacher and distill the student</strong></summary>

```bash
# Teacher — dense full-body tracking (PPO, 4096 envs). Output: output/teacher/g1_teacher/nn/
scripts/train_teacher.sh

# Student — multimodal distillation from the included teacher checkpoint.
scripts/train_student.sh
NUM_GPUS=4 scripts/train_student_multigpu.sh          # torchrun, single node
```

The teacher uses `InterAct/OMOMO_retarget_aug/` by default. Pass `--motion_file <directory>` to train on another G1 motion directory.

</details>

<details>
<summary><strong>5 · Finetune the student</strong></summary>

Start RL finetuning from a distilled student checkpoint:

```bash
scripts/train_finetune.sh <student.pth>
```

The finetuning configuration uses `InterAct/OMOMO_retarget_aug/` and the included tracking teacher by default.

</details>

## 🎮 Inference

### 🧠 Tracking teacher

The included checkpoint at `ultra/weights/teacher_ultra_inference.pth` is used for student distillation and inference. See the [model card](docs/teacher-model-card.md) for its training data and use terms.

For the released checkpoint, we further trained on AMASS and BONES-SEED alongside OMOMO after the original work. The training code in this repository follows the original OMOMO setting.

<details>
<summary>Run teacher inference</summary>

```bash
# Put one retargeted G1 .pt clip in /absolute/path/to/one_motion_dir.
python ultra/run_teacher_inference.py --motion-dir /absolute/path/to/one_motion_dir
```

The command writes `output/teacher_inference/rollout.pt` with observed/reference states and actions from the motion. Use `--checkpoint <teacher.pth>` for another tracking teacher or change `env.teacherPolicy` in `ultra/data/cfg/g1_student_vae.yaml` for distillation.

</details>

### 🤖 Student

```bash
scripts/play_student.sh <student.pth> 1 full_track      # Isaac Sim playback (also: sparse_track, object_obs points)
scripts/sim2sim_student.sh <student.pth> InterAct/OMOMO_retarget_aug full_track   # MuJoCo
scripts/export_jit.sh <student.pth> student_jit.pt      # TorchScript export
```

For a sampled, headless tracking success estimate (humanoid and humanoid + object):

```bash
scripts/eval_student.sh models/g1_student_vae_00005500.pth 16 100
```

The arguments after the checkpoint are the parallel environment count and completed episode count. The evaluator
uses `full_track`, seed 42, and the student configuration's domain randomization. It counts an episode as a humanoid
success if it does not terminate early and the mean global distance over 39 body origins stays below 0.3 m at every
evaluated step; combined success additionally requires object-center position error below 0.3 m at every step.
Episodes use the existing training sampler, including random starting frames and rollouts of at most 300 steps; they
do not systematically cover every clip. Use `--motion_file <directory>` after the episode count to select another
motion set, or `--seed <number>` to change the seed. This is an approximate tracking evaluation, not a reproduction
of Table I: the original ID/OOD splits, exact body/joint definition, and Isaac Gym evaluation protocol are not provided
here, and this repository uses Isaac Sim.

To evaluate each motion in a directory once from its first frame and write a Table I-style report:

```bash
scripts/eval_table1.sh models/g1_student_vae_00005500.pth InterAct/OMOMO_retarget_aug 16 output/table1_metrics.json
```

To evaluate a privileged teacher using the same tracking conditions, pass its checkpoint and select the teacher policy:

```bash
scripts/eval_table1.sh ultra/weights/teacher_ultra_inference.pth InterAct/OMOMO_retarget_aug \
  1024 output/teacher_table1.json --eval_policy teacher
```

The teacher receives its unmasked normalized privileged observation and uses its mean action. Its weights are loaded
by the environment; the student network constructed by the evaluation runner does not select actions.

The script evaluates the supplied motion directory without creating or inferring an ID/OOD split. The environment
count must be at least the number of distinct object type/scale combinations in the motion directory. Loading a large
motion set can require substantial GPU memory. A JSON report contains settings and one result per clip; the terminal
prints a summary. Runs use full tracking, seed 42, and the environment YAML's physics, randomization and
sensing settings unchanged. Student actions use the mean decoder output with a sampled prior latent.
Use `--seed <number>` after the output path to repeat with another seed. Append `--eval_randomization none` to disable physical randomization, reset perturbations, action delay,
observation noise and point-cloud/camera augmentation for this evaluation only. The default is `config`. ID/OOD describes the motion/object-scale split; toggling randomization does not create
an ID or OOD split. Evaluating the complete training directory is a full-corpus tracking check. No synthetic
OOD set is generated or claimed to match the paper's unavailable evaluation split.

The reported quantities are an **approximation**, not official Table I results. The two success rates use 0.3 m
thresholds at every frame, including the initial reference frame. Body errors use 39 body origins; the local
error subtracts each root position without rotational alignment. Object rotation is quaternion geodesic angle in
degrees. Jitter is the magnitude of the third finite difference of world body positions in m/s³. Errors are averaged
over simulated frames, including failed clips only up to termination. The 20 training padding frames are excluded.
The paper does not specify enough details here to verify exact joint sets, jitter, aggregation, or evaluation splits;
its reported experiments use Isaac Gym, whereas this repository uses Isaac Sim. The script does not calculate the
paper's reported uncertainty from repeated independent runs.

For a diagnostic closer to the student **training environment**, evaluate each supplied clip once using training's
hybrid random-start reset, 300-step rollout limit, stand-still sampling, domain randomization and observation-mask
curriculum at the checkpoint's saved epoch:

```bash
scripts/eval_training_conditions.sh models/g1_student_vae_00005500.pth InterAct/OMOMO_retarget_aug 16 output/training_prior.json prior
scripts/eval_training_conditions.sh models/g1_student_vae_00005500.pth InterAct/OMOMO_retarget_aug 16 output/training_encoder.json encoder
```

`prior` tests the deployable student; `encoder` is a privileged diagnostic using teacher observations, as in the
training encoder path. Both use mean student actions and do not mix in teacher actions, so neither reproduces the
training rollout policy exactly. The per-rank curriculum step counter is estimated as saved epoch times configured
horizon because it is not saved in the checkpoint. Clips start at random frames and may include the loader's 20
padding frames; each clip is sampled only once, so this is neither full-trajectory coverage nor the paper's Table I.
Random draws and resets can diverge between the two policy runs even with the same seed. The success metric retains
the approximate 0.3 m conditions described above.

The tracking evaluation scripts also accept checkpoints produced by `scripts/train_finetune.sh`. They automatically
detect the V3 finetune architecture, load its complete FiLM adapter and separate critic with strict weight matching,
and select `g1_student_finetune.yaml` for the network. The simulation still uses the same V2 tracking task and
`g1_student_vae.yaml` environment as distilled students, so it measures reference tracking rather than the V3
training task's perturbed-goal success. The optional `--eval_model finetune` asserts the detected architecture;
custom `--cfg_train` files are preserved.

```bash
# Same tracking-environment diagnostic as the distilled student; deployed prior policy.
scripts/eval_training_conditions.sh \
  models/g1_student_finetune_00006000.pth InterAct/OMOMO_retarget_aug \
  1024 output/finetune_prior_6000.json prior

# All supplied clips from frame zero, with the full_track observation preset.
scripts/eval_table1.sh \
  models/g1_student_finetune_00006000.pth InterAct/OMOMO_retarget_aug \
  1024 output/finetune_table1_6000.json
```

Here `1024` is the parallel environment count; every clip in the supplied directory is evaluated once. For the
training-environment diagnostic, the mask curriculum uses the saved checkpoint epoch and the estimated step counter
uses that epoch times the selected training configuration's horizon (16 for finetune, 8 for distillation). Both
use the tracking environment's 300-step rollout cap and its observation-mask probabilities. The JSON records the
model type, environment/network configs, horizon and protocol alongside the approximate success rates. Use a new
output filename for each checkpoint and policy.

## 📄 License

ULTRA's own code is released under Apache-2.0 (top-level `LICENSE`); the InterMimic-derived simulation and learning
stack is under MIT (`LICENSE-InterMimic`). `ultra/env/tasks/base_task.py` and `vec_task.py` are adapted from
NVIDIA's IsaacGym examples and keep NVIDIA's header. The simulator backend (`ultra/env/sim/`) uses Isaac Sim and
Isaac Lab, which are installed separately under their own licenses. Robot assets are from Unitree; motion data derives from OMOMO
through InterAct.

## 📖 Citation

```bibtex
@inproceedings{he2026ultra,
  title     = {ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation},
  author    = {He, Xialin and Xu, Sirui and Li, Xinyao and Dong, Runpei and Bian, Liuyu and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
