#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --job-name=square-dsrl
#SBATCH --output=logs/square_dsrl_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd third_party/dsrl
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120

python train_dsrl.py \
  --config-path=cfg/robomimic \
  --config-name=dsrl_square_comparison.yaml \
  seed="${SEED}" \
  total_timesteps=10000000 \
  eval_interval=100000 \
  num_evals=100 \
  wandb.project=square-online-rl-saturation
