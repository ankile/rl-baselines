# Harness Layout

- `experiments/`: canonical experiments (task, seeds, budgets, wandb metadata).
- `baselines/<id>/adapter.yaml`: baseline-specific launch/task/logging mapping.
- `envs/`: env setup specs and optional lockfiles.
- `slurm/profiles/`: resource presets per cluster.
- `slurm/templates/`: sbatch template(s).
- `tracking/upstreams.yaml`: upstream remotes, pinned commits, tracked paths, patch stack.
- `tracking/patches/<id>/`: patch artifacts applied on bootstrap.
- `tools/benchctl.py`: orchestration CLI.
