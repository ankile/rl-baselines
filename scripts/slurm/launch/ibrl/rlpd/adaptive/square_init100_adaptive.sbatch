#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --job-name=rlpd_sq_adaptive
#SBATCH --output=logs/rlpd_sq_adaptive_%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

SEEDS=(1 2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd third_party/ibrl
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120

# Make the adaptive sampler importable from self-improving-robots
export PYTHONPATH="${PYTHONPATH:-}:/iris/u/ankile/self-improving-robots-workspace/self-improving-robots"

python train_rl.py \
  --config_path release/cfgs/robomimic_rl/square_state_rlpd_adaptive.yaml \
  --seed="${SEED}" \
  --num_train_step 10000000 \
  --num_eval_episode 100 \
  --log_per_step 25000
