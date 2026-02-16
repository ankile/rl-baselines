#!/usr/bin/env python
"""Render robosuite NutAssemblySquare at specific (y, yaw) configurations.

Must be run with the ibrl Python environment (robosuite 1.4).
Usage:
    /iris/u/ankile/envs/ibrl/bin/python scripts/analysis/render_candidates.py
"""
from __future__ import annotations

import os
import sys

os.environ["MUJOCO_GL"] = "egl"

IBRL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ibrl")
IBRL_DIR = os.path.abspath(IBRL_DIR)
sys.path.insert(0, IBRL_DIR)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.robosuite_wrapper import PixelRobosuite


def yaw_to_qpos(y: float, yaw: float, x: float = -0.1125) -> np.ndarray:
    """Convert (y, yaw) to 7D object qpos [x, y, z, qw, qx, qy, qz]."""
    z = 0.8428  # approximate table-top height for the nut
    qw = np.cos(yaw / 2)
    qz = np.sin(yaw / 2)
    return np.array([x, y, z, qw, 0.0, 0.0, qz])


def main():
    env = PixelRobosuite(
        env_name="NutAssemblySquare",
        robots=["Panda"],
        episode_length=200,
        reward_shaping=False,
        image_size=256,
        rl_image_size=84,
        camera_names=["agentview", "robot0_eye_in_hand"],
        device="cuda",
    )

    # Candidates: (label, y, yaw_degrees)
    candidates = [
        ("hardest_interior_y0.116_yaw86", 0.1158, 86.4),
        ("yaw285_center_y", 0.168, 285.0),
        ("yaw90_center_y", 0.168, 90.0),
        ("easy_yaw0_center_y", 0.168, 0.0),
        ("easy_yaw180_center_y", 0.168, 180.0),
    ]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "renders")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(candidates), figsize=(5 * len(candidates), 5))

    for idx, (label, cy, cyaw_deg) in enumerate(candidates):
        cyaw = np.radians(cyaw_deg)
        qpos = yaw_to_qpos(cy, cyaw)
        _, high_res = env.reset_to_object_qpos(qpos)
        actual_x, actual_y, actual_yaw = env.get_nut_initial_state()

        img = high_res["agentview"].squeeze().permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(f"{label}\ny={actual_y:.4f}, yaw={np.degrees(actual_yaw):.0f}°", fontsize=9)
        ax.axis("off")

        # Also save individual
        individual_path = os.path.join(out_dir, f"{label}.png")
        fig_i, ax_i = plt.subplots(figsize=(4, 4))
        ax_i.imshow(img)
        ax_i.set_title(f"y={actual_y:.4f}, yaw={np.degrees(actual_yaw):.0f}°")
        ax_i.axis("off")
        fig_i.savefig(individual_path, dpi=100, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  Saved {individual_path}")

    combined_path = os.path.join(out_dir, "all_candidates.png")
    fig.tight_layout()
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined: {combined_path}")

    env.env.close()


if __name__ == "__main__":
    main()
