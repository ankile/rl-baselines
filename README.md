# rl-baselines

Unified harness for running and comparing RL baselines (`EXPO`, `ibrl`, `dsrl`, `qam`) with:

- One canonical experiment spec.
- Baseline adapters for command/env/task differences.
- Checked-in SLURM launch scripts (no generated submission layer).
- Upstream pinning + patch-based local delta tracking.

## Operating Policy (Project-Specific)

This repository is a single-user research harness for `ankile`, not a general-purpose framework.

- Prefer straightforward workflows over elaborate abstraction.
- Prefer fail-fast behavior over permissive fallbacks.
- If setup/runtime state is wrong, commands should error loudly with actionable messages.
- Optimize for getting reproducible runs working quickly on target clusters.

## Repository Model

This repo is the source of truth for orchestration and local modifications.
Upstream baseline repos are cloned into `third_party/` and treated as read-only upstream + local patch overlays.

## New Cluster Runbook (Exact Order)

1. Clone this repo:

```bash
git clone git@github.com:ankile/rl-baselines.git
cd rl-baselines
```

2. Run host checks:

```bash
./benchctl doctor
```

3. Bootstrap upstream baseline repos (clone/fetch/checkout/submodule/patch).
   Repos are placed under `third_party/`:

```bash
./benchctl bootstrap
```

4. Create baseline environments from the env specs (required for actual runs):

```bash
./benchctl bootstrap --create-envs
```

   Note: EXPO env creation auto-installs legacy MuJoCo 2.1.0 into `~/.mujoco/mujoco210` if missing. On offline nodes, pre-place that directory before running bootstrap.

5. Download pretrained checkpoints and data required by baselines:

   **ibrl** — Download the release data and models from [Google Drive](https://drive.google.com/file/d/1F2yH84Iqv0qRPmfH8o-kSzgtfaoqMzWE/view?usp=sharing), then extract into `third_party/ibrl/release/`. After extraction the directory should contain `release/data/` and `release/model/` alongside the existing `release/cfgs/`.

   ```bash
   # Download with gdown (pip install gdown if needed)
   gdown --fuzzy "https://drive.google.com/file/d/1F2yH84Iqv0qRPmfH8o-kSzgtfaoqMzWE/view?usp=sharing" -O /tmp/ibrl_release.zip
   unzip /tmp/ibrl_release.zip -d third_party/ibrl/release/
   ```

   **dsrl** — Download the published pretrained diffusion checkpoint and normalization stats from Google Drive:

   ```bash
   cd third_party/dsrl/dppo

   # Pretrained checkpoint
   mkdir -p log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint
   micromamba run -n dsrl gdown --fuzzy \
     "https://drive.google.com/file/d/1lP9mNe2AxMigfOywcaHOOR7FxQ-KR_Ee/view?usp=drive_link" \
     -O log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt

   # Normalization statistics
   mkdir -p log/robomimic/square
   micromamba run -n dsrl gdown --fuzzy \
     "https://drive.google.com/file/d/1_75UM0frCZVtcROgfWsdJ0FstToZd1b5/view?usp=drive_link" \
     -O log/robomimic/square/normalization.npz

   cd ../../..
   ```

   **expo** — Download the robomimic Square PH low-dim dataset expected by EXPO at `third_party/EXPO/robomimic/datasets/square/ph/low_dim_v141.hdf5`:

   ```bash
   if [ ! -f third_party/EXPO/robomimic/datasets/square/ph/low_dim_v141.hdf5 ]; then
     micromamba run -n expo python -m robomimic.scripts.download_datasets \
       --download_dir third_party/EXPO/robomimic/datasets \
       --tasks square \
       --dataset_types ph \
       --hdf5_types low_dim
   else
     echo "EXPO dataset already present at third_party/EXPO/robomimic/datasets/square/ph/low_dim_v141.hdf5"
   fi
   ```

   **qam** — Download the robomimic Square MH low-dim dataset expected by QAM at `~/.robomimic/square/mh/low_dim_v141.hdf5`:

   ```bash
   if [ ! -f ~/.robomimic/square/mh/low_dim_v141.hdf5 ]; then
     micromamba run -n qam python -m robomimic.scripts.download_datasets \
       --download_dir ~/.robomimic \
       --tasks square \
       --dataset_types mh \
       --hdf5_types low_dim
   else
     echo "QAM dataset already present at ~/.robomimic/square/mh/low_dim_v141.hdf5"
   fi
   ```

6. Validate harness config for the canonical Square benchmark:

```bash
./benchctl validate --experiment bench/experiments/square_online_rl.yaml
```

7. Submit jobs from checked-in scripts:

```bash
sbatch scripts/slurm/launch/expo/reproduce/square_online.sbatch
sbatch scripts/slurm/launch/qam/reproduce/square_online.sbatch
sbatch scripts/slurm/launch/dsrl/reproduce/square_online.sbatch
sbatch scripts/slurm/launch/ibrl/ibrl/reproduce/square_online.sbatch
sbatch scripts/slurm/launch/ibrl/rlpd/reproduce/square_online.sbatch
```

## Tracking Workflow

Refresh tracked patch files from local baseline repos:

```bash
./benchctl tracking refresh-patch --baseline expo
./benchctl tracking refresh-patch --baseline ibrl
./benchctl tracking refresh-patch --baseline dsrl
./benchctl tracking refresh-patch --baseline qam
```

Check pin/diff/patch state:

```bash
./benchctl tracking status
```

## Common Commands

Bootstrap one baseline only:

```bash
./benchctl bootstrap --baseline ibrl
```

Regenerate env lock metadata:

```bash
./benchctl lock-envs
```

## Direct Sbatch Scripts

Canonical launch scripts are checked in under `scripts/slurm/launch/` and organized by baseline and experiment type:

```bash
scripts/slurm/launch/expo/reproduce/square_online.sbatch
scripts/slurm/launch/qam/reproduce/square_online.sbatch
scripts/slurm/launch/dsrl/reproduce/square_online.sbatch
scripts/slurm/launch/ibrl/ibrl/reproduce/square_online.sbatch
scripts/slurm/launch/ibrl/rlpd/reproduce/square_online.sbatch
scripts/slurm/launch/ibrl/rlpd/init_scale/square_init10.sbatch
scripts/slurm/launch/ibrl/rlpd/init_scale/square_init50.sbatch
scripts/slurm/launch/ibrl/rlpd/init_scale/square_init80.sbatch
scripts/slurm/launch/ibrl/rlpd/init_scale/square_init100.sbatch
```

Example usage:

```bash
sbatch scripts/slurm/launch/ibrl/rlpd/init_scale/square_init10.sbatch
```

Launch assumptions (intentionally simple):
- Run `sbatch` from the `rl-baselines` repository root.
- Activate the correct micromamba/conda environment before launching.
- `third_party/<baseline>` repos and required data already exist from setup/bootstrap.
- If `third_party/ibrl` is stale after pulling updates, rerun `./benchctl bootstrap --baseline ibrl --force`.

Compatibility root-level scripts (`square_*.sbatch`) are still available.

## GitHub Setup

Create the public repo under your account (if not already created):

```bash
gh repo create ankile/rl-baselines --public --source=. --remote=origin --push
```

If `gh` auth fails, run:

```bash
gh auth login -h github.com
```

## Prerequisites

- Required: `git`, `python`, `zsh`, `micromamba`, `sbatch` (for launch), `curl`, `tar`.
- Optional: `gh` for GitHub automation.
- Note: this harness assumes `zsh` and executes setup commands under `zsh -lic`.

## Key Paths

- `bench/experiments/`: canonical benchmark specs.
- `bench/baselines/`: one adapter per baseline.
- `bench/slurm/`: cluster profiles and template.
- `scripts/slurm/launch/`: canonical checked-in sbatch launch scripts.
- `bench/tracking/`: upstream pins and patch files.
- `bench/tools/benchctl.py`: CLI entrypoint.
- `third_party/`: cloned upstream baseline repos managed by `benchctl bootstrap`.
