"""
Smooth success-probability surface from rollout data using Nadaraya-Watson
kernel regression with circular-aware distance for yaw.

Identifies the hardest initial state that is sufficiently interior to the
data distribution (so a 10%-span sub-distribution can be centered there).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Kernel regression helpers
# ---------------------------------------------------------------------------

def circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest angular distance on [0, 2pi), result in [0, pi]."""
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)


def nadaraya_watson(
    y_data: np.ndarray,
    yaw_data: np.ndarray,
    success_data: np.ndarray,
    y_query: np.ndarray,
    yaw_query: np.ndarray,
    h: float,
    y_scale: float,
    yaw_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-smoothed success probability and effective sample size.

    Returns
    -------
    prob : (len(yaw_query), len(y_query))
    n_eff : (len(yaw_query), len(y_query))  — Kish's effective sample size
    """
    n_yaw, n_y = len(yaw_query), len(y_query)
    prob = np.empty((n_yaw, n_y))
    n_eff = np.empty((n_yaw, n_y))
    for i, yq in enumerate(yaw_query):
        d_yaw = circular_distance(yaw_data, yq) / yaw_scale
        for j, yv in enumerate(y_query):
            d_y = np.abs(y_data - yv) / y_scale
            d2 = d_y**2 + d_yaw**2
            w = np.exp(-d2 / (2 * h**2))
            w_sum = w.sum()
            if w_sum < 1e-12:
                prob[i, j] = np.nan
                n_eff[i, j] = 0
            else:
                prob[i, j] = (w * success_data).sum() / w_sum
                n_eff[i, j] = w_sum**2 / (w**2).sum()  # Kish's n_eff
    return prob, n_eff


def loocv_log_likelihood(
    y_data: np.ndarray,
    yaw_data: np.ndarray,
    success_data: np.ndarray,
    h: float,
    y_scale: float,
    yaw_scale: float,
) -> float:
    """Leave-one-out cross-validated log-likelihood for bandwidth h."""
    N = len(y_data)
    ll = 0.0
    eps = 1e-10
    for k in range(N):
        d_yaw = circular_distance(yaw_data, yaw_data[k]) / yaw_scale
        d_y = np.abs(y_data - y_data[k]) / y_scale
        d2 = d_y**2 + d_yaw**2
        w = np.exp(-d2 / (2 * h**2))
        w[k] = 0.0  # leave one out
        w_sum = w.sum()
        if w_sum < 1e-12:
            continue
        p = (w * success_data).sum() / w_sum
        p = np.clip(p, eps, 1 - eps)
        ll += np.log(p) if success_data[k] else np.log(1 - p)
    return ll


def wilson_ci(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportion."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smooth success-probability surface")
    parser.add_argument("--input", required=True, help="Path to rollout pickle file")
    parser.add_argument("--output", default="results/success_surface.png")
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument(
        "--span-fraction", type=float, default=0.10,
        help="Fraction of original range for the sub-distribution (default: 0.10)",
    )
    parser.add_argument("--render", action="store_true", help="Render env at candidate states")
    args = parser.parse_args()

    # --- Load data ---
    with open(args.input, "rb") as f:
        records = pickle.load(f)

    y_data = np.array([r["nut_y"] for r in records])
    yaw_data = np.array([r["nut_yaw"] for r in records])
    success_data = np.array([r["success"] for r in records], dtype=float)
    fail_mask = ~success_data.astype(bool)

    N = len(records)
    y_min, y_max = y_data.min(), y_data.max()
    y_range = y_max - y_min

    print(f"Loaded {N} episodes")
    print(f"  y   range: [{y_min:.4f}, {y_max:.4f}] (span={y_range:.4f})")
    print(f"  yaw range: [{yaw_data.min():.4f}, {yaw_data.max():.4f}]")
    print(f"  success rate: {success_data.mean():.3f} ({int(success_data.sum())}/{N})")

    # --- Normalisation scales ---
    y_scale = y_range
    yaw_scale = np.pi  # max circular distance

    # --- Interior constraints for sub-distribution ---
    half_span_y = args.span_fraction * y_range / 2
    y_interior_lo = y_min + half_span_y
    y_interior_hi = y_max - half_span_y
    print(f"\n--- Interior constraints (span_fraction={args.span_fraction}) ---")
    print(f"  10% y span: {2*half_span_y:.5f}  (half={half_span_y:.5f})")
    print(f"  Feasible y center: [{y_interior_lo:.5f}, {y_interior_hi:.5f}]")
    print(f"  yaw: circular — no boundary constraint")

    # --- Bandwidth selection via LOOCV ---
    bandwidths = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    print("\nLOOCV bandwidth selection:")
    best_h, best_ll = None, -np.inf
    for h in bandwidths:
        ll = loocv_log_likelihood(y_data, yaw_data, success_data, h, y_scale, yaw_scale)
        tag = ""
        if ll > best_ll:
            best_ll = ll
            best_h = h
            tag = " <-- best"
        print(f"  h={h:.2f}  log-lik={ll:.2f}{tag}")
    assert best_h is not None
    print(f"Selected bandwidth: h={best_h}")

    # --- Evaluate on grid ---
    G = args.grid_size
    y_grid = np.linspace(y_min, y_max, G)
    yaw_grid = np.linspace(0, 2 * np.pi, G, endpoint=False)

    print(f"\nEvaluating on {G}x{G} grid...")
    prob, n_eff = nadaraya_watson(
        y_data, yaw_data, success_data, y_grid, yaw_grid, best_h, y_scale, yaw_scale
    )
    ci_lo, ci_hi = wilson_ci(prob, n_eff)

    # --- Find global minimum ---
    min_idx = np.unravel_index(np.nanargmin(prob), prob.shape)
    print(f"\nGlobal hardest point (unconstrained):")
    print(f"  y={y_grid[min_idx[1]]:.4f}, yaw={np.degrees(yaw_grid[min_idx[0]]):.1f}°, "
          f"p={prob[min_idx]:.3f}, n_eff={n_eff[min_idx]:.0f}, "
          f"CI=[{ci_lo[min_idx]:.3f}, {ci_hi[min_idx]:.3f}]")
    print(f"  y percentile: {(y_data < y_grid[min_idx[1]]).mean()*100:.1f}%")

    # --- Find hardest INTERIOR point ---
    interior_mask = np.ones_like(prob, dtype=bool)
    for j, yv in enumerate(y_grid):
        if yv < y_interior_lo or yv > y_interior_hi:
            interior_mask[:, j] = False

    prob_interior = prob.copy()
    prob_interior[~interior_mask] = np.nan

    int_min_idx = np.unravel_index(np.nanargmin(prob_interior), prob_interior.shape)
    int_min_yaw = yaw_grid[int_min_idx[0]]
    int_min_y = y_grid[int_min_idx[1]]
    int_min_p = prob[int_min_idx]
    int_min_neff = n_eff[int_min_idx]

    print(f"\nHardest INTERIOR point (y in [{y_interior_lo:.4f}, {y_interior_hi:.4f}]):")
    print(f"  y   = {int_min_y:.4f}")
    print(f"  yaw = {int_min_yaw:.4f} rad = {np.degrees(int_min_yaw):.1f}°")
    print(f"  p(success) = {int_min_p:.3f}")
    print(f"  n_eff = {int_min_neff:.0f}")
    print(f"  CI  = [{ci_lo[int_min_idx]:.3f}, {ci_hi[int_min_idx]:.3f}]")
    print(f"  y percentile: {(y_data < int_min_y).mean()*100:.1f}%")

    # Top-10 interior hardest
    prob_flat = prob_interior.ravel()
    order = np.argsort(prob_flat)
    print(f"\nTop-10 hardest interior grid cells:")
    for rank, fi in enumerate(order[:10]):
        iy, jy = np.unravel_index(fi, prob.shape)
        if np.isnan(prob_flat[fi]):
            break
        print(
            f"  #{rank+1}: y={y_grid[jy]:.4f}, "
            f"yaw={np.degrees(yaw_grid[iy]):.1f}°, "
            f"p={prob[iy, jy]:.3f}, n_eff={n_eff[iy, jy]:.0f}, "
            f"CI=[{ci_lo[iy, jy]:.3f}, {ci_hi[iy, jy]:.3f}]"
        )

    # --- Raw binned validation near best interior point ---
    print(f"\n--- Raw data validation near interior minimum ---")
    for radius_pct in [5, 10, 15, 20, 25]:
        r_y = y_range * radius_pct / 100
        r_yaw = 2 * np.pi * radius_pct / 100
        mask = (np.abs(y_data - int_min_y) < r_y) & (circular_distance(yaw_data, int_min_yaw) < r_yaw)
        n_pts = mask.sum()
        if n_pts > 0:
            sr = success_data[mask].mean()
            n_fail = int((~success_data[mask].astype(bool)).sum())
            print(f"  {radius_pct}% radius: {n_pts:4d} pts, {n_fail:3d} failures, success={sr:.3f}")

    # --- Plot ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaw_deg = np.degrees(yaw_grid)
    int_min_yaw_deg = np.degrees(int_min_yaw)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # Helper to draw the 10% span box for the interior minimum
    def draw_span_box(ax, center_y, center_yaw_deg, color="cyan", label="10% span"):
        span_yaw_deg = np.degrees(args.span_fraction * 2 * np.pi)
        rect = Rectangle(
            (center_y - half_span_y, center_yaw_deg - span_yaw_deg / 2),
            2 * half_span_y, span_yaw_deg,
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--", label=label,
        )
        ax.add_patch(rect)

    # (a) Success probability heatmap
    ax = axes[0, 0]
    pcm = ax.pcolormesh(y_grid, yaw_deg, prob, shading="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    fig.colorbar(pcm, ax=ax, label="p(success)")
    ax.scatter(y_data[fail_mask], np.degrees(yaw_data[fail_mask]), c="black", s=3, alpha=0.3, label="failures")
    # Interior bounds
    ax.axvline(y_interior_lo, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.axvline(y_interior_hi, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    # Best interior point
    ax.plot(int_min_y, int_min_yaw_deg, "w*", markersize=15, markeredgecolor="black", label=f"hardest interior")
    draw_span_box(ax, int_min_y, int_min_yaw_deg)
    ax.set_xlabel("nut_y")
    ax.set_ylabel("nut_yaw (deg)")
    ax.set_title("Success probability")
    ax.legend(loc="upper right", fontsize=7)

    # (b) Effective sample size
    ax = axes[0, 1]
    pcm2 = ax.pcolormesh(y_grid, yaw_deg, n_eff, shading="auto", cmap="viridis")
    fig.colorbar(pcm2, ax=ax, label="n_eff")
    ax.axvline(y_interior_lo, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.axvline(y_interior_hi, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.plot(int_min_y, int_min_yaw_deg, "w*", markersize=15, markeredgecolor="black")
    ax.set_xlabel("nut_y")
    ax.set_ylabel("nut_yaw (deg)")
    ax.set_title("Effective sample size (data density)")

    # (c) Confidence interval width
    ax = axes[0, 2]
    ci_width = ci_hi - ci_lo
    pcm3 = ax.pcolormesh(y_grid, yaw_deg, ci_width, shading="auto", cmap="magma_r")
    fig.colorbar(pcm3, ax=ax, label="CI width (95%)")
    ax.axvline(y_interior_lo, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.axvline(y_interior_hi, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.plot(int_min_y, int_min_yaw_deg, "w*", markersize=15, markeredgecolor="black")
    ax.set_xlabel("nut_y")
    ax.set_ylabel("nut_yaw (deg)")
    ax.set_title("95% CI width (uncertainty)")

    # (d) Data scatter with density
    ax = axes[1, 0]
    ax.scatter(y_data[~fail_mask], np.degrees(yaw_data[~fail_mask]), c="green", s=2, alpha=0.15, label="success")
    ax.scatter(y_data[fail_mask], np.degrees(yaw_data[fail_mask]), c="red", s=12, alpha=0.7, label="failure", zorder=5)
    ax.axvline(y_interior_lo, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8, label="interior bounds")
    ax.axvline(y_interior_hi, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.plot(int_min_y, int_min_yaw_deg, "k*", markersize=15, label="hardest interior")
    draw_span_box(ax, int_min_y, int_min_yaw_deg, color="blue", label="10% span")
    ax.set_xlabel("nut_y")
    ax.set_ylabel("nut_yaw (deg)")
    ax.set_title("All data points")
    ax.legend(loc="upper right", fontsize=7)

    # (e) 1D marginal: success vs yaw
    ax = axes[1, 1]
    marginal_yaw = np.nanmean(prob, axis=1)
    marginal_yaw_interior = np.nanmean(prob_interior, axis=1)
    ax.plot(yaw_deg, marginal_yaw, "b-", linewidth=2, label="all y")
    ax.plot(yaw_deg, marginal_yaw_interior, "r-", linewidth=2, alpha=0.7, label="interior y only")
    ax.axhline(success_data.mean(), color="gray", linestyle="--", alpha=0.5, label="overall mean")
    ax.axvline(int_min_yaw_deg, color="red", linestyle=":", alpha=0.6, label=f"hardest yaw={int_min_yaw_deg:.0f}°")
    ax.set_xlabel("nut_yaw (deg)")
    ax.set_ylabel("p(success)")
    ax.set_title("Marginal: success vs yaw")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7)

    # (f) 1D marginal: success vs y
    ax = axes[1, 2]
    marginal_y = np.nanmean(prob, axis=0)
    ax.plot(y_grid, marginal_y, "b-", linewidth=2)
    ax.axhline(success_data.mean(), color="gray", linestyle="--", alpha=0.5, label="overall mean")
    ax.axvline(y_interior_lo, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8, label="interior bounds")
    ax.axvline(y_interior_hi, color="cyan", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.axvline(int_min_y, color="red", linestyle=":", alpha=0.6, label=f"hardest y={int_min_y:.4f}")
    ax.set_xlabel("nut_y")
    ax.set_ylabel("p(success)")
    ax.set_title("Marginal: success vs y")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Success surface (N={N}, h={best_h}, grid={G}x{G}) | "
        f"Best interior: y={int_min_y:.4f}, yaw={int_min_yaw_deg:.0f}deg, p={int_min_p:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")

    # --- Optional: render environment at candidate states ---
    if args.render:
        render_candidates(args, int_min_y, int_min_yaw, out_path)


def render_candidates(args, target_y: float, target_yaw: float, out_path: Path):
    """Render the environment at the hardest interior point + a few comparison states."""
    import os
    import sys

    os.environ["MUJOCO_GL"] = "egl"
    IBRL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ibrl")
    IBRL_DIR = os.path.abspath(IBRL_DIR)
    sys.path.insert(0, IBRL_DIR)

    import torch
    from env.robosuite_wrapper import PixelRobosuite

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

    candidates = [
        ("hardest_interior", target_y, target_yaw),
        ("easy_reference", 0.168, 0.0),  # near center, yaw=0
        ("yaw_270", 0.168, np.radians(285)),  # center y, hard yaw
    ]

    render_dir = out_path.parent / "renders"
    render_dir.mkdir(exist_ok=True)

    for label, cy, cyaw in candidates:
        # Convert (y, yaw) to 7D qpos: [x, y, z, qw, qx, qy, qz]
        x = -0.1125  # near-constant from data
        z = 0.8428   # table height (nut resting on table)
        qw = float(np.cos(cyaw / 2))
        qz = float(np.sin(cyaw / 2))
        object_qpos = np.array([x, cy, z, qw, 0.0, 0.0, qz])

        obs, high_res = env.reset_to_object_qpos(object_qpos)
        actual_x, actual_y, actual_yaw = env.get_nut_initial_state()

        # Grab the high-res agentview image
        img = high_res["agentview"].squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        fig_r, ax_r = plt.subplots(figsize=(4, 4))
        ax_r.imshow(img)
        ax_r.set_title(f"{label}\ny={actual_y:.4f}, yaw={np.degrees(actual_yaw):.0f}deg")
        ax_r.axis("off")
        fname = render_dir / f"{label}.png"
        fig_r.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig_r)
        print(f"Rendered {label} -> {fname}")

    env.env.close()


if __name__ == "__main__":
    main()
