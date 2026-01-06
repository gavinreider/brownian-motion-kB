# Brownian Motion Analysis of Boltzmann's Constant

These python scripts detect 1 micron diameter beads in microscope images and videos, track their Brownian motion, and estimate Boltzmann's constant. Data and plots that are generated are organized in `data/` and `pictures-videos/`.

## Layout
- `code/` - analysis scripts
  - `track_movement.py` - processes all `.avi` files in `pictures-videos/Brownian Motion Videos`, detects and connects trajectories, and saves `data/tracks_<video>.csv` and `pictures-videos/trajectories_<video>.png`.
  - `trajectory_velocities.py` - reads `data/tracks_*.csv`, calculates per-trajectory velocities, residual RMS, diffusion coefficient, and kB, then writes `data/velocities_<video>.csv` and `data/summary_kB.csv`.
  - `count_beads.py` - counts beads in still depth images from `pictures-videos/Background and Varying Depth Photos`, saving `data/particle_counts_vs_depth.csv` and `pictures-videos/bead_count_plot.png`.
  - `estimate_kB.py` - fits ln(count) vs depth from `data/particle_counts_vs_depth.csv` to estimate kB and saves `pictures-videos/ln_count_fit.png`.
- `data/` - CSV outputs (tracks, velocities, summary_kB, bead counts).
- `pictures-videos/` - raw images/videos and generated plots.

## Requirements
Install dependencies from `code/requirements.txt`:
```bash
pip install -r code/requirements.txt
```

## Usage
Track videos:
   ```bash
   cd code
   python track_movement.py
   ```
   This outputs per-video tracks (`data/tracks_<video>.csv`) and trajectory plots (`pictures-videos/trajectories_<video>.png`).

Compute velocities and kB from tracks:
   ```bash
   python trajectory_velocities.py
   ```
   This produces `data/velocities_<video>.csv` and `data/summary_kB.csv` with kB estimates for each video.

Static depth series:
   ```bash
   python count_beads.py
   python estimate_kB.py
   ```
   This generates `data/particle_counts_vs_depth.csv`, `pictures-videos/bead_count_plot.png`, and `pictures-videos/ln_count_fit.png`.

## Notes on large files
The raw `.avi` videos from microscope observation exceed GitHub’s 100 MB limit. Keep them locally or track them with Git LFS. If you do not use LFS, remember to keep the `.gitignore` entry that excludes `pictures-videos/Brownian Motion Videos/*.avi`.
