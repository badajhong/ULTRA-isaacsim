#!/usr/bin/env bash
# Usage: eval_table1.sh CHECKPOINT MOTION_DIRECTORY [NUM_ENVS] [OUTPUT_JSON] [extra arguments...]
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:?Usage: scripts/eval_table1.sh CHECKPOINT MOTION_DIRECTORY [NUM_ENVS] [OUTPUT_JSON]}"
motion_dir="${2:?Specify the directory of ID or OOD motion clips}"
num_envs="${3:-16}"
output="${4:-output/table1_metrics.json}"
if (($# > 4)); then shift 4; else set --; fi
if ! [[ "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer." >&2
    exit 2
fi
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
python ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --test --headless --task_mode full_track --seed 42 --num_envs "$num_envs" \
    --checkpoint "$checkpoint" --motion_file "$motion_dir" --eval_episodes 1 \
    --eval_full_clips --eval_output "$output" "$@"
