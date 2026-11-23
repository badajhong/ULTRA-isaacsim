#!/usr/bin/env bash
# Sample tracking episodes. Usage: eval_student.sh CHECKPOINT [NUM_ENVS] [EPISODES] [extra run_distill.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:?Usage: scripts/eval_student.sh CHECKPOINT [NUM_ENVS] [EPISODES] [extra args...]}"
shift
num_envs="${1:-16}"
if (($#)); then shift; fi
episodes="${1:-100}"
if (($#)); then shift; fi
if ! [[ "$num_envs" =~ ^[1-9][0-9]*$ && "$episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS and EPISODES must be positive integers." >&2
    exit 2
fi
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
python ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --test --headless --task_mode full_track --seed 42 \
    --num_envs "$num_envs" --checkpoint "$checkpoint" --eval_episodes "$episodes" "$@"
