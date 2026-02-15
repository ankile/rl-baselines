#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_init_s1
#SBATCH --output=logs/eval_init_s1_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

export MUJOCO_GL=egl

python scripts/eval_initial_states.py \
  --artifact self-improving/rlpd-square-convergence/model-rlpd_square_init1_s1:v0 \
  --num_episodes 5000 \
  --seed 0 \
  --output results/init_states_conv_s1_n5000.pkl
