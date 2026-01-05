"""
Track bead trajectories for all videos and save per-video tracks/plots.

Outputs per video:
  - data/tracks_<video>.csv
  - pictures-videos/trajectories_<video>.png

Run trajectory_velocities.py afterwards to compute velocities and kB.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # avoid GUI windows/popups
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp

# -----------------------------------------------------------
# Parameters
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "pictures-videos" / "Brownian Motion Videos"
OUTPUT_DATA_DIR = BASE_DIR / "data"
OUTPUT_PICS_DIR = BASE_DIR / "pictures-videos"
VIDEO_GLOB = "*.avi"
N_FRAMES = None           # number of frames to process (set None for all)
diameter = 9             # bead size in px
minmass = 120            # raise to reject weak noise; lower if missing beads
separation = int(diameter * 1.3)  # minimum spacing between peaks
search_range = 6         # max displacement (px) per frame
memory = 1               # frames to remember a lost particle
min_track_length = 20    # drop tracks shorter than this many frames
fps = 34.722             # video frame rate, used to express velocities in px/s


def preprocess(im):
    """Grayscale + blur + gentle bandpass."""
    g = np.asarray(im)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    g = tp.bandpass(g, lshort=1, llong=15)
    return g


def dedupe_close(feats, radius):
    """Keep only the strongest detection within `radius` pixels (per frame)."""
    if feats is None or len(feats) == 0:
        return feats
    feats = feats.sort_values("mass", ascending=False).copy()
    keep = []
    taken = np.zeros(len(feats), dtype=bool)
    coords = feats[["x", "y"]].values
    for i, (x, y) in enumerate(coords):
        if taken[i]:
            continue
        keep.append(i)
        dx = coords[:, 0] - x
        dy = coords[:, 1] - y
        taken |= (dx * dx + dy * dy) < (radius * radius)
    return feats.iloc[keep]


def process_video(video_path: Path):
    raw_frames = pims.Video(str(video_path))
    frames = pims.pipeline(preprocess)(raw_frames)
    frame_slice = slice(None) if N_FRAMES is None else slice(0, N_FRAMES)

    feats_list = []
    for i, frame in enumerate(frames[frame_slice]):
        f = tp.locate(
            frame,
            diameter=diameter,
            minmass=minmass,
            separation=separation,
            noise_size=1,
            smoothing_size=diameter,
            threshold=0.01,
        )
        if f is None or len(f) == 0:
            continue
        f["frame"] = i
        feats_list.append(f)
    if not feats_list:
        print(f"No features found in {video_path.name}; try lowering minmass or threshold.")
        return
    feats = pd.concat(feats_list, ignore_index=True)

    feats = feats.groupby("frame", group_keys=False).apply(lambda df: dedupe_close(df, radius=diameter))
    feats = feats.reset_index(drop=True)

    linked = tp.link_df(feats, search_range=search_range, memory=memory)
    linked = tp.filter_stubs(linked, threshold=min_track_length)

    if len(linked) == 0:
        print(f"No linked trajectories after filtering for {video_path.name}; skipping outputs.")
        return

    # Plot trajectories
    first_img = preprocess(raw_frames[0])
    tp.plot_traj(linked, superimpose=first_img, label=False)
    plt.title(f"Trajectories: {video_path.name}")
    OUTPUT_PICS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PICS_DIR / f"trajectories_{video_path.stem}.png", dpi=200, bbox_inches="tight")
    plt.close()

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tracks_out = OUTPUT_DATA_DIR / f"tracks_{video_path.stem}.csv"
    linked.to_csv(tracks_out, index=False)
    print(f"Saved {tracks_out} ({linked['particle'].nunique()} tracks)")


def main():
    videos = sorted(VIDEO_DIR.glob(VIDEO_GLOB))
    if not videos:
        raise FileNotFoundError(f"No videos matching {VIDEO_GLOB} in {VIDEO_DIR}")
    for vid in videos:
        print(f"Processing {vid.name} ...")
        process_video(vid)


if __name__ == "__main__":
    main()
