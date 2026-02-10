#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --job-name=square-qam
#SBATCH --output=logs/square_qam_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
mkdir -p logs/.numba_cache

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd third_party/qam
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120
export NUMBA_CACHE_DIR="${SLURM_SUBMIT_DIR:-$PWD}/logs/.numba_cache"

python main.py \
  --run_group=qam-square \
  --wandb_project=square-online-rl-saturation \
  --agent=agents/qam.py \
  --tags=QAM \
  --seed="${SEED}" \
  --env_name=square-mh-low_dim \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --offline_steps=1000000 \
  --online_steps=10000000 \
  --utd_ratio=1 \
  --eval_episodes=100 \
  --eval_interval=100000 \
  --log_interval=5000 \
  --save_interval=100000 \
  --agent.num_qs=10 \
  --agent.discount=0.99 \
  --agent.batch_size=256 \
  --agent.rho=0.5 \
  --agent.inv_temp=1.0
