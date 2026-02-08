# rl-baselines

Unified harness for running and comparing RL baselines (`EXPO`, `ibrl`, `dsrl`, `qam`) with:

- One canonical experiment spec.
- Baseline adapters for command/env/task differences.
- Shared SLURM rendering/launch flow.
- Upstream pinning + patch-based local delta tracking.

## Repository Model

This repo is the source of truth for orchestration and local modifications.
Upstream baseline repos remain external/read-only dependencies.

## Quick Start

1. Clone this repo and ensure sibling baseline repos exist (or let bootstrap clone them).
2. Run a health check:

```bash
python bench/tools/benchctl.py doctor
```

3. Validate the canonical Square experiment:

```bash
python bench/tools/benchctl.py validate --experiment bench/experiments/square_online_rl.yaml
```

4. Render SLURM jobs:

```bash
python bench/tools/benchctl.py render --experiment bench/experiments/square_online_rl.yaml
```

5. Launch (dry-run first):

```bash
python bench/tools/benchctl.py launch --experiment bench/experiments/square_online_rl.yaml --dry-run
```

## GitHub Setup

Create the public repo under your account (if not already created):

```bash
gh repo create ankile/rl-baselines --public --source=. --remote=origin --push
```

If `gh` auth fails, run:

```bash
gh auth login -h github.com
```

## Key Paths

- `bench/experiments/`: canonical benchmark specs.
- `bench/baselines/`: one adapter per baseline.
- `bench/slurm/`: cluster profiles and template.
- `bench/tracking/`: upstream pins and patch files.
- `bench/tools/benchctl.py`: CLI entrypoint.
