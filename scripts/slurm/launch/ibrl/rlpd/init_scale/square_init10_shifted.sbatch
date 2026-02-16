#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=iris
#SBATCH --array=0-4
#SBATCH --job-name=rlpd_sq_i10sh
#SBATCH --output=logs/rlpd_sq_init10_shifted_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

SEEDS=(1 2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SCALE=0.1

# Region A (shifted-hardest): center at x=-0.1125, y=0.11575, yaw=86 deg
# Effective ranges: x=[-0.11275,-0.11225], y=[0.110,0.1215], yaw=[68,104] deg
CENTER="-0.1125,0.11575,86.0"

cd third_party/ibrl
export PATH="/iris/u/ankile/envs/ibrl/bin:$PATH"
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120

python train_rl.py \
  --config_path release/cfgs/robomimic_rl/square_state_rlpd.yaml \
  --seed="${SEED}" \
  --square_init_scale "${SCALE}" \
  --square_init_center="${CENTER}" \
  --run_name_method rlpd \
  --wandb_project rlpd-square-initscale \
  --num_train_step 10000000 \
  --num_eval_episode 100 \
  --log_per_step 25000 \
  --log_eval_video 1
