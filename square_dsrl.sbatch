#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --job-name=square-dsrl
#SBATCH --output=logs/square_dsrl_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${ROOT}/logs"

if [ -d "${ROOT}/third_party/dsrl" ]; then
  DSRL_DIR="${ROOT}/third_party/dsrl"
else
  echo "ERROR: Missing '${ROOT}/third_party/dsrl'. Run './benchctl bootstrap --baseline dsrl' from '${ROOT}'." >&2
  exit 1
fi

micromamba run -n dsrl bash -lc "
set -euo pipefail
cd '${DSRL_DIR}'
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120
python train_dsrl.py \
  --config-path=cfg/robomimic \
  --config-name=dsrl_square_comparison.yaml \
  seed=${SEED} \
  total_timesteps=10000000 \
  eval_interval=100000 \
  num_evals=100 \
  wandb.project=square-online-rl-saturation
"
