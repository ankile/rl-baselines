# rl-baselines

Unified harness for running and comparing RL baselines (`EXPO`, `ibrl`, `dsrl`, `qam`) with:

- One canonical experiment spec.
- Baseline adapters for command/env/task differences.
- Shared SLURM rendering/launch flow.
- Upstream pinning + patch-based local delta tracking.

## Repository Model

This repo is the source of truth for orchestration and local modifications.
Upstream baseline repos remain external/read-only dependencies.

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

3. Bootstrap upstream baseline repos (clone/fetch/checkout/submodule/patch):

```bash
./benchctl bootstrap
```

4. (Optional) Create baseline environments from the env specs:

```bash
./benchctl bootstrap --create-envs
```

5. Validate harness config for the canonical Square benchmark:

```bash
./benchctl validate --experiment bench/experiments/square_online_rl.yaml
```

6. Render generated sbatch files:

```bash
./benchctl render --experiment bench/experiments/square_online_rl.yaml
```

7. Verify launch commands (no submit):

```bash
./benchctl launch --experiment bench/experiments/square_online_rl.yaml --dry-run
```

8. Submit jobs:

```bash
./benchctl launch --experiment bench/experiments/square_online_rl.yaml
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

Render just one baseline:

```bash
./benchctl render --experiment bench/experiments/square_online_rl.yaml --baseline expo --show
```

Bootstrap one baseline only:

```bash
./benchctl bootstrap --baseline ibrl
```

Regenerate env lock metadata:

```bash
./benchctl lock-envs
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

## Prerequisites

- Required: `git`, `python`, `micromamba`, `sbatch` (for launch).
- Optional but recommended: `gh` for GitHub automation.
- Note: `benchctl doctor` checks shell-level command resolution and `gh auth status`.

## Key Paths

- `bench/experiments/`: canonical benchmark specs.
- `bench/baselines/`: one adapter per baseline.
- `bench/slurm/`: cluster profiles and template.
- `bench/tracking/`: upstream pins and patch files.
- `bench/tools/benchctl.py`: CLI entrypoint.
