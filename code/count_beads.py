from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "pictures-videos" / "Background and Varying Depth Photos"
DATA_DIR = BASE_DIR / "data"
PICS_DIR = BASE_DIR / "pictures-videos"

# Load images
frames = pims.ImageSequence(str(IMG_DIR / "*.png"))
background_raw = frames[0]   # background image


# Preprocessing function tuned for images
def preprocess(img, bg):
    def to_gray(a):
        arr = np.array(a)
        # If image is RGB/RGBA, convert to single-channel grayscale
        if arr.ndim == 3:
            if arr.shape[2] == 4:  # drop alpha if present
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected image shape {arr.shape}, expected 2D or 3-channel.")
        return arr.astype(np.uint8)

    img = to_gray(img)
    bg  = to_gray(bg)

    # Light Gaussian smoothing to reduce noise
    img_blur = cv2.GaussianBlur(img, (5,5), 0)

    # Gentle background subtraction
    # (raw subtraction is too harsh for your faint beads)
    sub = cv2.subtract(img_blur, bg)

    # Trackpy bandpass (tuned)
    # lowpass=1 px, highpass=25 px
    filtered = tp.bandpass(sub, 1, 25)

    return filtered

# Trackpy detection parameters tuned for your beads
tp_params = dict(
    diameter=9,        # bead size in px
    minmass=5,         # Faintness sensitivity
    separation=7,      # avoid double detection
    noise_size=1,      # slight noise suppression
    smoothing_size=3,  
    threshold=0.005,
    characterize=True
)


# Analyze all sample frames
depths = np.arange(0, 205, 5)  # 0–200 µm in 5 µm steps
particle_counts = []

for i in range(1, len(frames)):
    depth = depths[i-1]

    proc = preprocess(frames[i], background_raw)
    f = tp.locate(proc, **tp_params)

    # Remove out-of-focus detections, keep only circular beads
    f = f[f['ecc'] < 0.6]

    # Keep only sharp beads (focused σ small)
    f = f[f['size'] < 4.0]

    particle_counts.append({
        "depth_um": depth,
        "count": len(f)
    })

    print(f"Depth {depth:3d} µm: {len(f)} beads")

# Save & plot
df = pd.DataFrame(particle_counts)
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_DIR / "particle_counts_vs_depth.csv", index=False)

plt.plot(df["depth_um"], df["count"], marker="o", linestyle="none")
plt.xlabel("Depth (µm)")
plt.ylabel("Particle Count")
plt.title("Bead Count vs Depth")
plt.grid(True)
PICS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(PICS_DIR / "bead_count_plot.png", dpi=200, bbox_inches="tight")
plt.close()
