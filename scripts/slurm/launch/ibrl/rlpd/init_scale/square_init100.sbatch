#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --job-name=rlpd_sq_init100
#SBATCH --output=logs/rlpd_sq_init100_%A_%a.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${ROOT}/benchctl" ]; then
  echo "ERROR: Launch this script from the rl-baselines repository root." >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"

if [ -d "${ROOT}/third_party/ibrl" ]; then
  IBRL_DIR="${ROOT}/third_party/ibrl"
else
  echo "ERROR: Missing '${ROOT}/third_party/ibrl'. Run './benchctl bootstrap --baseline ibrl' from '${ROOT}'." >&2
  exit 1
fi

SEEDS=(1 2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SCALE=1.0

micromamba run -n ibrl bash -lc "
set -euo pipefail
cd '${IBRL_DIR}'
export MUJOCO_GL=egl
python train_rl.py \
  --config_path release/cfgs/robomimic_rl/square_state_rlpd_initscale.yaml \
  --seed ${SEED} \
  --square_init_scale ${SCALE} \
  --wandb_project rlpd-square-initscale \
  --log_per_step 25000
"
