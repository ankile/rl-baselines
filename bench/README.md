# Harness Layout

- `experiments/`: canonical experiments (task, seeds, budgets, wandb metadata).
- `baselines/<id>/adapter.yaml`: baseline-specific task/env/logging metadata.
- `envs/`: env setup specs and lockfiles.
- `slurm/profiles/`: resource presets per cluster.
- `slurm/templates/`: sbatch template(s).
- `tracking/upstreams.yaml`: upstream remotes, pinned commits, tracked paths, patch stack.
- `tracking/patches/<id>/`: patch artifacts applied on bootstrap.
- `tools/benchctl.py`: orchestration CLI.
- `../scripts/slurm/launch/`: canonical checked-in sbatch launch scripts.
- `third_party/`: cloned upstream baseline repos (managed by `./benchctl bootstrap`).

## Core CLI

- `./benchctl doctor`
- `./benchctl bootstrap`
- `./benchctl bootstrap --create-envs`
- `./benchctl validate --experiment bench/experiments/square_online_rl.yaml`
- `sbatch scripts/slurm/launch/<baseline>/<family>/<script>.sbatch`

## Development Bias

- Keep this harness pragmatic and explicit.
- Avoid introducing optional flows unless they are actively needed in this project.
- On ambiguous or broken setup states, prefer hard failure over silent recovery.
- Runtime shell assumption: `zsh`.
