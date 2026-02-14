#!/usr/bin/env python
"""Roll out a trained RLPD policy and record initial states + outcomes.

Usage examples:

  # From a wandb artifact (downloads checkpoint automatically):
  python scripts/eval_initial_states.py \
      --artifact rlpd-square-convergence/model-rlpd_square_init1_s1:latest \
      --num_episodes 500 --seed 0 --output results/init_states_s1.pkl

  # From a local checkpoint:
  python scripts/eval_initial_states.py \
      --weight third_party/ibrl/exps/rl/.../latest.pt \
      --num_episodes 500 --seed 0 --output results/init_states_s1.pkl

Output (pickle): list of dicts, each with:
  - episode: int
  - seed: int (per-episode seed used for env.reset)
  - nut_x, nut_y, nut_yaw: float  (initial nut pose)
  - success: bool
  - episode_return: float
  - episode_length: int
"""

import argparse
import os
import sys
import pickle
import time

import numpy as np
import torch

# We need to run from inside third_party/ibrl for imports to resolve.
IBRL_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "ibrl")
IBRL_DIR = os.path.abspath(IBRL_DIR)
sys.path.insert(0, IBRL_DIR)

os.environ["MUJOCO_GL"] = "egl"


def download_artifact(artifact_name: str, project: str | None = None) -> str:
    """Download a wandb artifact and return the local directory path."""
    import wandb

    api = wandb.Api()
    if project and "/" not in artifact_name:
        artifact_name = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_name)
    path = artifact.download()
    print(f"Downloaded artifact to {path}")
    return path


def load_agent(weight_path: str, device: str = "cuda"):
    """Load agent + env params from a checkpoint directory."""
    import train_rl

    agent, _eval_env, eval_env_params = train_rl.load_model(weight_path, device)
    agent.eval()
    return agent, eval_env_params


def rollout(agent, env_params: dict, num_episodes: int, seed: int, verbose: bool = True) -> list[dict]:
    """Run episodes and collect initial states + outcomes."""
    from env.robosuite_wrapper import PixelRobosuite
    from common_utils import ibrl_utils as utils

    env = PixelRobosuite(**env_params)
    is_square = env_params.get("env_name") == "NutAssemblySquare"
    if not is_square:
        print("Warning: env is not NutAssemblySquare; initial states will not include nut pose.")

    records = []
    t0 = time.time()

    with torch.no_grad(), utils.eval_mode(agent):
        for ep in range(num_episodes):
            ep_seed = seed + ep
            np.random.seed(ep_seed)

            obs, _ = env.reset()

            initial_state = env.get_nut_initial_state() if is_square else None

            terminal = False
            while not terminal:
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminal, _, _ = env.step(action)

            success = env.episode_reward > 0
            rec = {
                "episode": ep,
                "seed": ep_seed,
                "success": bool(success),
                "episode_return": float(env.episode_reward),
                "episode_length": int(env.time_step),
            }
            if initial_state is not None:
                rec["nut_x"], rec["nut_y"], rec["nut_yaw"] = initial_state

            records.append(rec)

            if verbose and (ep + 1) % 50 == 0:
                sr = np.mean([r["success"] for r in records])
                elapsed = time.time() - t0
                eps_per_sec = (ep + 1) / elapsed
                eta = (num_episodes - ep - 1) / eps_per_sec
                print(
                    f"  [{ep+1}/{num_episodes}] "
                    f"success_rate={sr:.3f}  "
                    f"speed={eps_per_sec:.1f} ep/s  "
                    f"ETA={eta:.0f}s"
                )

    sr = np.mean([r["success"] for r in records])
    print(f"Done: {num_episodes} episodes, success_rate={sr:.3f}, time={time.time()-t0:.1f}s")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact", type=str, help="W&B artifact name (e.g. project/model-name:v0)")
    src.add_argument("--weight", type=str, help="Local path to checkpoint .pt file")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Output pickle path (default: auto-generated)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.artifact:
        artifact_dir = download_artifact(args.artifact)
        weight_path = os.path.join(artifact_dir, "latest.pt")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(artifact_dir, "model0.pt")
    else:
        weight_path = args.weight
    assert os.path.exists(weight_path), f"Checkpoint not found: {weight_path}"

    # Load agent
    agent, env_params = load_agent(weight_path, args.device)

    # Run rollouts
    records = rollout(agent, env_params, args.num_episodes, args.seed, verbose=not args.quiet)

    # Save results
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        tag = os.path.basename(os.path.dirname(weight_path))
        args.output = f"results/init_states_{tag}_n{args.num_episodes}_s{args.seed}.pkl"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} records to {args.output}")

    # Quick summary
    successes = [r["success"] for r in records]
    print(f"\nSummary:")
    print(f"  Episodes:     {len(records)}")
    print(f"  Success rate: {np.mean(successes):.3f}")
    if "nut_x" in records[0]:
        xs = [r["nut_x"] for r in records]
        ys = [r["nut_y"] for r in records]
        yaws = [r["nut_yaw"] for r in records]
        print(f"  Nut X range:  [{min(xs):.4f}, {max(xs):.4f}]")
        print(f"  Nut Y range:  [{min(ys):.4f}, {max(ys):.4f}]")
        print(f"  Nut yaw range:[{min(yaws):.4f}, {max(yaws):.4f}]")

        # Failure breakdown by quadrant
        fail_records = [r for r in records if not r["success"]]
        if fail_records:
            cx = np.mean(xs)
            cy = np.mean(ys)
            quads = {"top-right": 0, "top-left": 0, "bottom-left": 0, "bottom-right": 0}
            for r in fail_records:
                qx = "right" if r["nut_x"] >= cx else "left"
                qy = "top" if r["nut_y"] >= cy else "bottom"
                quads[f"{qy}-{qx}"] += 1
            print(f"  Failures by quadrant (center={cx:.4f},{cy:.4f}):")
            for q, n in quads.items():
                print(f"    {q}: {n}")


if __name__ == "__main__":
    main()
