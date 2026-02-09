#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --job-name=square-ibrl
#SBATCH --output=logs/square_ibrl_%A_%a.out

set -euo pipefail

ROOT="/iris/u/ankile/self-improving-robots-workspace/rl-baselines"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${ROOT}/logs"

micromamba run -n ibrl bash -lc "
set -euo pipefail
cd '${ROOT}/third_party/ibrl'
export MUJOCO_GL=egl
export WANDB_DIR=.
export WANDB__SERVICE_WAIT=120
export CPATH=\"\$CONDA_PREFIX/include\${CPATH:+:\$CPATH}\"
export LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}\"
for d in \"\$CONDA_PREFIX\"/lib/python*/site-packages/nvidia/*/lib; do
  [ -d \"\$d\" ] && export LD_LIBRARY_PATH=\"\$d\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"
done
python train_rl.py \
  --config_path release/cfgs/robomimic_rl/square_state_online.yaml \
  --seed=${SEED} \
  --run_name_method ibrl \
  --wandb_project square-online-rl-saturation \
  --wandb_entity ankile \
  --num_train_step 10000000 \
  --num_eval_episode 100 \
  --log_per_step 100000
"
