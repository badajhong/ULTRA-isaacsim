#!/usr/bin/env bash
# Usage: eval_training_conditions.sh CHECKPOINT MOTION_DIRECTORY [NUM_ENVS] [OUTPUT_JSON] [prior|encoder] [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:?Usage: scripts/eval_training_conditions.sh CHECKPOINT MOTION_DIRECTORY [NUM_ENVS] [OUTPUT_JSON] [prior|encoder]}"
motion_dir="${2:?Specify the motion directory}"
num_envs="${3:-16}"
output="${4:-output/training_conditions_metrics.json}"
policy="${5:-prior}"
if (($# > 5)); then shift 5; else set --; fi
if ! [[ "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer." >&2
    exit 2
fi
if [[ "$policy" != prior && "$policy" != encoder ]]; then
    echo "Policy must be prior or encoder." >&2
    exit 2
fi
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
python ultra/run_distill.py --task UltraDistillObjV2Point \
    --cfg_env ultra/data/cfg/g1_student_vae.yaml \
    --cfg_train ultra/data/cfg/train/rlg/g1_student_vae.yaml \
    --test --headless --seed 42 --num_envs "$num_envs" --checkpoint "$checkpoint" \
    --motion_file "$motion_dir" --eval_episodes 1 --eval_full_clips \
    --eval_training_conditions --eval_policy "$policy" --eval_output "$output" "$@"
