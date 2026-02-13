#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --job-name=square-expo
#SBATCH --output=logs/square_expo_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd third_party/EXPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120

python train_robo.py \
  --env_name=square \
  --seed="${SEED}" \
  --dataset_dir=ph \
  --utd_ratio=10 \
  --start_training=0 \
  --pretrain_steps=200000 \
  --pretrain_q=False \
  --pretrain_edit=False \
  --max_steps=10000000 \
  --eval_episodes=100 \
  --eval_interval=100000 \
  --config=configs/expo_config.py \
  --config.backup_entropy=False \
  --config.hidden_dims='(256, 256, 256)' \
  --config.num_min_qs=2 \
  --config.N=8 \
  --config.n_edit_samples=8 \
  --config.edit_action_scale=0.1 \
  --project_name=square-online-rl-saturation \
  --wandb_entity=ankile
