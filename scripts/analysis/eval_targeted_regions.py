#!/usr/bin/env python
"""Evaluate a trained policy on specific (x, y, yaw) sub-regions.

Defines two candidate 10%-span regions, runs rollouts from each, and
renders visualizations of the region centers and corners.

Must run with the ibrl Python env (robosuite 1.4):
    MUJOCO_GL=egl /iris/u/ankile/envs/ibrl/bin/python scripts/analysis/eval_targeted_regions.py \
        --weight artifacts/model-rlpd_square_init1_s1:v0/latest.pt \
        --num_episodes 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["MUJOCO_GL"] = "egl"

IBRL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ibrl")
IBRL_DIR = os.path.abspath(IBRL_DIR)
sys.path.insert(0, IBRL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---- Original distribution ranges (from 5000-episode data) ----
ORIG_X_MIN, ORIG_X_MAX = -0.115, -0.110
ORIG_Y_MIN, ORIG_Y_MAX = 0.110, 0.225
ORIG_YAW_MIN, ORIG_YAW_MAX = 0.0, 2 * np.pi

X_SPAN = ORIG_X_MAX - ORIG_X_MIN    # 0.005
Y_SPAN = ORIG_Y_MAX - ORIG_Y_MIN    # 0.115
YAW_SPAN = 2 * np.pi                # 6.283

# 10% sub-distribution half-spans
FRAC = 0.10
HALF_X = FRAC * X_SPAN / 2      # 0.00025
HALF_Y = FRAC * Y_SPAN / 2      # 0.00575
HALF_YAW = FRAC * YAW_SPAN / 2  # 0.31416 rad = 18 deg


def clamp_center(center: float, half: float, lo: float, hi: float) -> float:
    """Shift a center so that [center-half, center+half] fits inside [lo, hi]."""
    if center - half < lo:
        center = lo + half
    if center + half > hi:
        center = hi - half
    return center


def wrap_yaw_range(center_yaw: float, half: float):
    """Return (lo, hi) for a yaw range, handling wraparound via modular sampling."""
    lo = center_yaw - half
    hi = center_yaw + half
    return lo, hi  # we'll sample with modular arithmetic


def sample_from_region(region: dict, n: int, rng: np.random.RandomState) -> list[np.ndarray]:
    """Sample n initial states as 7D qpos arrays from a region dict."""
    samples = []
    for _ in range(n):
        x = rng.uniform(region["x_lo"], region["x_hi"])
        y = rng.uniform(region["y_lo"], region["y_hi"])
        # Sample yaw uniformly, handling wraparound
        yaw = rng.uniform(region["yaw_lo"], region["yaw_hi"])
        yaw = yaw % (2 * np.pi)

        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)
        z = 0.8428
        samples.append(np.array([x, y, z, qw, 0.0, 0.0, qz]))
    return samples


def define_regions() -> dict[str, dict]:
    """Define the two candidate regions."""
    # Region A: "shifted-hardest" — centered at hardest point, shifted inward
    hardest_x, hardest_y, hardest_yaw_deg = -0.1125, 0.1135, 86.0
    hardest_yaw = np.radians(hardest_yaw_deg)

    a_x = clamp_center(hardest_x, HALF_X, ORIG_X_MIN, ORIG_X_MAX)
    a_y = clamp_center(hardest_y, HALF_Y, ORIG_Y_MIN, ORIG_Y_MAX)
    # yaw is circular — no clamping needed
    a_yaw_lo, a_yaw_hi = wrap_yaw_range(hardest_yaw, HALF_YAW)

    region_a = dict(
        name="A: shifted-hardest (yaw~86°)",
        label="shifted_hardest",
        center_x=a_x, center_y=a_y, center_yaw=hardest_yaw,
        x_lo=a_x - HALF_X, x_hi=a_x + HALF_X,
        y_lo=a_y - HALF_Y, y_hi=a_y + HALF_Y,
        yaw_lo=a_yaw_lo, yaw_hi=a_yaw_hi,
    )

    # Region B: "interior-hard" — centered at yaw~285, y~0.168
    b_x, b_y, b_yaw_deg = -0.1125, 0.168, 285.0
    b_yaw = np.radians(b_yaw_deg)

    b_x = clamp_center(b_x, HALF_X, ORIG_X_MIN, ORIG_X_MAX)
    b_y = clamp_center(b_y, HALF_Y, ORIG_Y_MIN, ORIG_Y_MAX)
    b_yaw_lo, b_yaw_hi = wrap_yaw_range(b_yaw, HALF_YAW)

    region_b = dict(
        name="B: interior-hard (yaw~285°)",
        label="interior_hard",
        center_x=b_x, center_y=b_y, center_yaw=b_yaw,
        x_lo=b_x - HALF_X, x_hi=b_x + HALF_X,
        y_lo=b_y - HALF_Y, y_hi=b_y + HALF_Y,
        yaw_lo=b_yaw_lo, yaw_hi=b_yaw_hi,
    )

    return {"A": region_a, "B": region_b}


def print_region(r: dict):
    """Pretty-print a region's ranges."""
    print(f"  {r['name']}")
    print(f"    x:   [{r['x_lo']:.6f}, {r['x_hi']:.6f}]  center={r['center_x']:.6f}")
    print(f"    y:   [{r['y_lo']:.6f}, {r['y_hi']:.6f}]  center={r['center_y']:.6f}")
    yaw_lo_deg = np.degrees(r["yaw_lo"])
    yaw_hi_deg = np.degrees(r["yaw_hi"])
    center_deg = np.degrees(r["center_yaw"])
    print(f"    yaw: [{yaw_lo_deg:.1f}°, {yaw_hi_deg:.1f}°]  center={center_deg:.1f}°")
    print(f"         [{r['yaw_lo']:.4f}, {r['yaw_hi']:.4f}] rad")


def run_rollouts(agent, env, region: dict, num_episodes: int, seed: int) -> list[dict]:
    """Run rollouts sampling initial states from a region."""
    from common_utils import ibrl_utils as utils

    rng = np.random.RandomState(seed)
    qpos_list = sample_from_region(region, num_episodes, rng)
    records = []
    t0 = time.time()

    with torch.no_grad(), utils.eval_mode(agent):
        for ep, qpos in enumerate(qpos_list):
            obs, _ = env.reset_to_object_qpos(qpos)
            actual_x, actual_y, actual_yaw = env.get_nut_initial_state()

            terminal = False
            while not terminal:
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminal, _, _ = env.step(action)

            success = env.episode_reward > 0
            records.append({
                "episode": ep,
                "success": bool(success),
                "nut_x": actual_x,
                "nut_y": actual_y,
                "nut_yaw": actual_yaw,
                "episode_return": float(env.episode_reward),
                "episode_length": int(env.time_step),
            })

            if (ep + 1) % 50 == 0:
                sr = np.mean([r["success"] for r in records])
                elapsed = time.time() - t0
                eps = (ep + 1) / elapsed
                print(f"    [{ep+1}/{num_episodes}] sr={sr:.3f}  {eps:.1f} ep/s")

    sr = np.mean([r["success"] for r in records])
    print(f"    Done: {num_episodes} eps, success_rate={sr:.3f}")
    return records


def render_region(render_env, region: dict, out_dir: str):
    """Render the center and 4 yaw-y corners of a region."""
    label = region["label"]
    os.makedirs(out_dir, exist_ok=True)

    cx, cy, cyaw = region["center_x"], region["center_y"], region["center_yaw"]

    # Points to render: center + 4 corners (vary y and yaw at extremes)
    points = [
        ("center", cx, cy, cyaw),
        ("lo_y__lo_yaw", cx, region["y_lo"], region["yaw_lo"] % (2*np.pi)),
        ("lo_y__hi_yaw", cx, region["y_lo"], region["yaw_hi"] % (2*np.pi)),
        ("hi_y__lo_yaw", cx, region["y_hi"], region["yaw_lo"] % (2*np.pi)),
        ("hi_y__hi_yaw", cx, region["y_hi"], region["yaw_hi"] % (2*np.pi)),
    ]

    images = []
    titles = []
    for ptname, px, py, pyaw in points:
        qw = np.cos(pyaw / 2)
        qz = np.sin(pyaw / 2)
        qpos = np.array([px, py, 0.8428, qw, 0.0, 0.0, qz])
        _, high_res = render_env.reset_to_object_qpos(qpos)
        _, ay_, ayaw_ = render_env.get_nut_initial_state()

        # Use first available camera
        cam_key = list(high_res.keys())[0]
        img = high_res[cam_key].squeeze().permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        images.append(img)
        titles.append(f"{ptname}\ny={ay_:.4f} yaw={np.degrees(ayaw_):.0f}°")

    fig, axes = plt.subplots(1, len(points), figsize=(4 * len(points), 4.5))
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")
    fig.suptitle(f"Region {region['name']}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"region_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved renders: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/targeted_eval")
    parser.add_argument("--skip_rollouts", action="store_true", help="Only render, skip rollouts")
    args = parser.parse_args()

    import train_rl
    from env.robosuite_wrapper import PixelRobosuite

    # Load agent
    print("Loading agent...")
    agent, _eval_env, eval_env_params = train_rl.load_model(args.weight, args.device)
    agent.eval()

    # Create env
    print("Creating environment...")
    env = PixelRobosuite(**eval_env_params)

    # Define regions
    regions = define_regions()

    print("\n" + "=" * 60)
    print("CANDIDATE REGIONS (10% of original distribution span)")
    print("=" * 60)
    print(f"\nOriginal distribution:")
    print(f"  x:   [{ORIG_X_MIN}, {ORIG_X_MAX}]  span={X_SPAN}")
    print(f"  y:   [{ORIG_Y_MIN}, {ORIG_Y_MAX}]  span={Y_SPAN:.4f}")
    print(f"  yaw: [0°, 360°]  span=360°")
    print(f"\n10% sub-distribution spans:")
    print(f"  x:   {FRAC*X_SPAN:.4f}  (half={HALF_X:.5f})")
    print(f"  y:   {FRAC*Y_SPAN:.4f}  (half={HALF_Y:.5f})")
    print(f"  yaw: {np.degrees(FRAC*YAW_SPAN):.1f}°  (half={np.degrees(HALF_YAW):.1f}°)")
    print()
    for key, region in regions.items():
        print_region(region)
        print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Render regions (separate env with agentview camera for nice visuals)
    print("=" * 60)
    print("RENDERING REGIONS")
    print("=" * 60)
    render_env = PixelRobosuite(
        env_name="NutAssemblySquare",
        robots=["Panda"],
        episode_length=200,
        reward_shaping=False,
        image_size=256,
        rl_image_size=96,
        camera_names=["agentview"],
        device=args.device,
    )
    render_dir = os.path.join(args.output_dir, "renders")
    for key, region in regions.items():
        print(f"\nRegion {key}:")
        render_region(render_env, region, render_dir)
    render_env.env.close()

    # Run rollouts
    if not args.skip_rollouts:
        print("\n" + "=" * 60)
        print(f"RUNNING ROLLOUTS ({args.num_episodes} episodes per region)")
        print("=" * 60)

        import pickle
        all_results = {}
        for key, region in regions.items():
            print(f"\nRegion {key}: {region['name']}")
            records = run_rollouts(agent, env, region, args.num_episodes, args.seed)
            all_results[key] = records

            # Save per-region results
            pkl_path = os.path.join(args.output_dir, f"rollouts_{region['label']}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(records, f)
            print(f"  Saved: {pkl_path}")

        # Summary comparison
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for key, region in regions.items():
            records = all_results[key]
            successes = [r["success"] for r in records]
            sr = np.mean(successes)
            n_fail = sum(1 for s in successes if not s)

            # Wilson CI
            n = len(records)
            z = 1.96
            denom = 1 + z**2 / n
            centre = (sr + z**2 / (2 * n)) / denom
            half = z * np.sqrt(sr * (1 - sr) / n + z**2 / (4 * n**2)) / denom
            ci_lo = max(0, centre - half)
            ci_hi = min(1, centre + half)

            print(f"\n  {region['name']}:")
            print(f"    Success: {int(sum(successes))}/{n} = {sr:.3f}")
            print(f"    Failures: {n_fail}")
            print(f"    95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

            # Breakdown: actual sampled ranges
            ys = [r["nut_y"] for r in records]
            yaws = [np.degrees(r["nut_yaw"]) for r in records]
            print(f"    Actual y range:   [{min(ys):.4f}, {max(ys):.4f}]")
            print(f"    Actual yaw range: [{min(yaws):.1f}°, {max(yaws):.1f}°]")

        # Comparison with kernel model predictions
        print("\n  Kernel model predictions (for reference):")
        print("    Region A (shifted-hardest, yaw~86°): ~0.64")
        print("    Region B (interior-hard, yaw~285°):  ~0.84")

    env.env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
