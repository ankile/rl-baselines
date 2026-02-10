#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --job-name=rlpd_sq_init80
#SBATCH --output=logs/rlpd_sq_init80_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

SEEDS=(1 2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SCALE=0.8

cd third_party/ibrl
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120

if ! python train_rl.py --help 2>&1 | grep -q -- "--square_init_scale"; then
  echo "ERROR: init-scale support missing in third_party/ibrl (train_rl.py has no --square_init_scale)." >&2
  echo "Run from rl-baselines root: ./benchctl bootstrap --baseline ibrl --force" >&2
  exit 2
fi

python train_rl.py \
  --config_path release/cfgs/robomimic_rl/square_state_rlpd.yaml \
  --seed="${SEED}" \
  --square_init_scale "${SCALE}" \
  --run_name_method rlpd \
  --wandb_project rlpd-square-initscale \
  --wandb_entity ankile \
  --num_train_step 10000000 \
  --num_eval_episode 100 \
  --log_eval_video 1 \
  --num_eval_video_episode 10 \
  --log_per_step 25000
