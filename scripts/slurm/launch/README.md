# SLURM Launch Scripts

These scripts are the canonical, checked-in job launch entrypoints.

Assumptions:
- Run `sbatch` from the `rl-baselines` repository root.
- Activate the correct environment before launch.
- Required `third_party/<baseline>` source trees and data already exist.

Layout:
- `expo/reproduce/`: EXPO square online RL scripts.
- `qam/reproduce/`: QAM square online RL scripts.
- `dsrl/reproduce/`: DSRL square online RL scripts.
- `ibrl/ibrl/reproduce/`: IBRL square online RL scripts.
- `ibrl/rlpd/reproduce/`: RLPD square online RL scripts.
- `ibrl/rlpd/init_scale/`: RLPD square init-scale sweeps.
