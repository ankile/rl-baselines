#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --job-name=square-qam
#SBATCH --output=logs/square_qam_%A_%a.out

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${ROOT}/logs"

if [ -d "${ROOT}/third_party/qam" ]; then
  QAM_DIR="${ROOT}/third_party/qam"
else
  echo "ERROR: Missing '${ROOT}/third_party/qam'. Run './benchctl bootstrap --baseline qam' from '${ROOT}'." >&2
  exit 1
fi

micromamba run -n qam bash -lc "
set -euo pipefail
cd '${QAM_DIR}'
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120
export CPATH=\"\$CONDA_PREFIX/include\${CPATH:+:\$CPATH}\"
export LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}\"
for d in \"\$CONDA_PREFIX\"/lib/python*/site-packages/nvidia/*/lib; do
  [ -d \"\$d\" ] && export LD_LIBRARY_PATH=\"\$d\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"
done
python main.py \
  --run_group=qam-square \
  --wandb_project=square-online-rl-saturation \
  --agent=agents/qam.py \
  --tags=QAM \
  --seed=${SEED} \
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
"
