"""
Post-process per-video tracks to compute velocities, residual RMS, and kB.

Looks for tracks_<video>.csv files (from track_movement.py) and writes:
  - velocities_<video>.csv with per-trajectory stats + weighted mean row
  - summary_kB.csv with one row per video
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Constants / parameters
PIXELS_PER_MICRON = 4  # 4 px = 1 µm
FPS = 34.722           # frames per second
ETA = 1.0e-3           # Pa·s
BEAD_RADIUS_M = 0.5e-6 # meters
TEMP_K = 296.15         # Kelvin
TRACKS_GLOB = "tracks_*.csv"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def compute_velocity_and_kb(linked_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Compute per-trajectory stats, residual RMS, D, and kB estimate."""
    linked_df = linked_df.sort_values(["particle", "frame"]).reset_index(drop=True)
    dframe = linked_df.groupby("particle")["frame"].diff()
    dx = linked_df.groupby("particle")["x"].diff()
    dy = linked_df.groupby("particle")["y"].diff()

    mask = dframe.notna() & (dframe != 0)
    steps = linked_df.loc[mask].copy()
    steps["vx_px_per_frame"] = dx[mask] / dframe[mask]
    steps["vy_px_per_frame"] = dy[mask] / dframe[mask]
    steps["speed_px_per_frame"] = np.hypot(steps["vx_px_per_frame"], steps["vy_px_per_frame"])

    summary = steps.groupby("particle").agg(
        mean_vx_px_per_frame=("vx_px_per_frame", "mean"),
        mean_vy_px_per_frame=("vy_px_per_frame", "mean"),
        mean_speed_px_per_frame=("speed_px_per_frame", "mean"),
        n_steps=("speed_px_per_frame", "size"),
    )

    # Weighted global mean velocity vector
    total_steps = summary["n_steps"].sum()
    w_mean_vx_px_per_frame = (summary["mean_vx_px_per_frame"] * summary["n_steps"]).sum() / total_steps
    w_mean_vy_px_per_frame = (summary["mean_vy_px_per_frame"] * summary["n_steps"]).sum() / total_steps

    # Residuals after subtracting weighted mean
    steps["vx_res_px_per_frame"] = steps["vx_px_per_frame"] - w_mean_vx_px_per_frame
    steps["vy_res_px_per_frame"] = steps["vy_px_per_frame"] - w_mean_vy_px_per_frame
    steps["speed_res_px_per_frame"] = np.hypot(
        steps["vx_res_px_per_frame"], steps["vy_res_px_per_frame"]
    )

    rms_residual = steps.groupby("particle").apply(
        lambda df: np.sqrt(np.mean(df["speed_res_px_per_frame"] ** 2))
    ).rename("rms_residual_speed_px_per_frame")
    summary = summary.join(rms_residual)

    # Units conversions
    summary["mean_vx_px_per_s"] = summary["mean_vx_px_per_frame"] * FPS
    summary["mean_vy_px_per_s"] = summary["mean_vy_px_per_frame"] * FPS
    summary["mean_speed_px_per_s"] = summary["mean_speed_px_per_frame"] * FPS
    summary["mean_vx_um_per_s"] = summary["mean_vx_px_per_s"] / PIXELS_PER_MICRON
    summary["mean_vy_um_per_s"] = summary["mean_vy_px_per_s"] / PIXELS_PER_MICRON
    summary["mean_speed_um_per_s"] = summary["mean_speed_px_per_s"] / PIXELS_PER_MICRON
    summary["rms_residual_speed_px_per_s"] = summary["rms_residual_speed_px_per_frame"] * FPS
    summary["rms_residual_speed_um_per_s"] = (
        summary["rms_residual_speed_px_per_s"] / PIXELS_PER_MICRON
    )

    # Weighted average RMS residual speed across trajectories
    avg_rms_residual_px_per_frame = (
        (summary["rms_residual_speed_px_per_frame"] * summary["n_steps"]).sum() / total_steps
    )
    avg_rms_residual_um_per_s = avg_rms_residual_px_per_frame * FPS / PIXELS_PER_MICRON

    # Diffusion coefficient and kB estimate
    dt = 1.0 / FPS
    D_um2_per_s = (avg_rms_residual_um_per_s ** 2) * dt / 4.0
    D_m2_per_s = D_um2_per_s * 1e-12
    k_B_est = (6 * np.pi * ETA * BEAD_RADIUS_M * D_m2_per_s) / TEMP_K
    k_B_ratio = k_B_est / 1.380649e-23

    weighted_row = pd.DataFrame(
        {
            "particle": ["weighted_mean"],
            "mean_vx_px_per_frame": [w_mean_vx_px_per_frame],
            "mean_vy_px_per_frame": [w_mean_vy_px_per_frame],
            "mean_speed_px_per_frame": [np.hypot(w_mean_vx_px_per_frame, w_mean_vy_px_per_frame)],
            "n_steps": [total_steps],
            "mean_vx_px_per_s": [w_mean_vx_px_per_frame * FPS],
            "mean_vy_px_per_s": [w_mean_vy_px_per_frame * FPS],
            "mean_speed_px_per_s": [np.hypot(w_mean_vx_px_per_frame, w_mean_vy_px_per_frame) * FPS],
            "mean_vx_um_per_s": [w_mean_vx_px_per_frame * FPS / PIXELS_PER_MICRON],
            "mean_vy_um_per_s": [w_mean_vy_px_per_frame * FPS / PIXELS_PER_MICRON],
            "mean_speed_um_per_s": [
                np.hypot(w_mean_vx_px_per_frame, w_mean_vy_px_per_frame) * FPS / PIXELS_PER_MICRON
            ],
            "rms_residual_speed_px_per_frame": [avg_rms_residual_px_per_frame],
            "rms_residual_speed_px_per_s": [avg_rms_residual_px_per_frame * FPS],
            "rms_residual_speed_um_per_s": [avg_rms_residual_um_per_s],
            "D_m2_per_s": [D_m2_per_s],
            "k_B_est_J_per_K": [k_B_est],
            "k_B_ratio_to_accepted": [k_B_ratio],
        }
    )

    out = pd.concat([summary.reset_index(), weighted_row], ignore_index=True)
    return out, k_B_est, k_B_ratio


def main():
    tracks_files = sorted(DATA_DIR.glob(TRACKS_GLOB))
    if not tracks_files:
        raise FileNotFoundError(f"No tracks files matching {TRACKS_GLOB} in {DATA_DIR}")

    summary_rows = []
    for tracks_file in tracks_files:
        df = pd.read_csv(tracks_file)
        if not {"particle", "frame", "x", "y"}.issubset(df.columns):
            print(f"Skipping {tracks_file.name}: missing required columns")
            continue
        base = tracks_file.stem.replace("tracks_", "")
        print(f"Processing {tracks_file.name} ...")
        velocities, kB_est, kB_ratio = compute_velocity_and_kb(df)
        vel_out = DATA_DIR / f"velocities_{base}.csv"
        velocities.to_csv(vel_out, index=False)
        summary_rows.append(
            {
                "video": base,
                "tracks_file": tracks_file.name,
                "velocities_file": vel_out.name,
                "k_B_est_J_per_K": kB_est,
                "k_B_ratio_to_accepted": kB_ratio,
            }
        )
        print(
            f"  k_B = {kB_est:.3e} J/K "
            f"({kB_ratio:.3f}× accepted); wrote {vel_out.name}"
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(DATA_DIR / "summary_kB.csv", index=False)
        print("Wrote summary_kB.csv")


if __name__ == "__main__":
    main()
