# SAUVC AUV — Visual Odometry + Sensor Fusion

Pipeline for **SAUVC 2026** pool navigation: SuperPoint/LightGlue feature matching, RANSAC motion estimation, quality gating, and EKF fusion with IMU + pressure depth.

## Is your approach sound?

**Yes — this is a strong foundation for SAUVC**, with a few caveats:

| Strength | Why it helps |
|----------|--------------|
| Tile intersections | High-contrast, repeatable corners — ideal for SuperPoint |
| Known tile spacing | Gives **metric scale** to monocular VO (critical) |
| IMU prediction | Fills gaps when visual is rejected (motion blur, low texture) |
| Pressure depth | Stable Z estimate; keeps altitude above pool floor (avoid 5 pt penalty / auto-abort) |
| RANSAC + quality gate | Prevents bad matches from corrupting the EKF |

**Watch out for:**

1. **Yaw drift** — IMU gyro integrates drift over ~16 m navigation leg. Use visual yaw updates and/or align to tile grid direction.
2. **Compute** — SuperPoint+LightGlue needs a **Jetson Orin/NX** or similar GPU for 30 FPS. ORB fallback is included for bring-up.
3. **Bottom contact** — SAUVC penalizes floor touches (5 pts) and aborts after 10 s cumulative. Keep depth controlled; VO assumes a flat floor.
4. **This is localization, not the full stack** — You still need **gate detection**, **drum color classification**, and **flare detection** as separate modules for Tasks 1–4.
5. **Calibrate** — Camera intrinsics, IMU axes, and surface pressure must be measured in your practice pool.

## Architecture

```
IMU ──► EKF predict ──► state [x,y,z,vx,vy,vz,yaw]
                              ▲
Pressure ── depth update ─────┤
                              │
Video ── SuperPoint/LightGlue ── RANSAC ── quality gate ── visual update
```

## Setup

```bash
cd /Users/macbook/Documents/VO
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: LightGlue (GPU recommended)
pip install git+https://github.com/cvg/LightGlue.git
```

## Run your five videos

The video runner accepts either individual paths or every MP4 in a directory.
It writes one output folder per video containing `trajectory.png` and
`frame_log.csv`, plus a combined `batch_summary.csv`.

```bash
# Default dataset location (does not copy the videos)
python scripts/run_vo_on_video.py \
  --output /Users/macbook/Documents/VO/output/dataset_run \
  --frame-stride 2

# With the configured defaults, this is equivalent:
python scripts/run_vo_on_video.py --output /Users/macbook/Documents/VO/output/dataset_run
```

Use `frame_log.csv` to inspect rejected visual updates. A trajectory alone is
not evidence that the odometry works: check the acceptance rate, inlier ratio,
and reprojection error before trusting a run.

## Quick demo (your sample tile frame)

```bash
python scripts/demo_on_sample.py --synthetic-shift
# Output: output/matches.png, output/summary.png
```

## Run on your 5 tile videos

Place videos in `data/videos/` then:

```bash
python scripts/run_vo_on_video.py \
  --video data/videos/tile_run_01.mp4 \
  --imu data/imu.csv \
  --pressure data/pressure.csv
```

### EfficientLoFTR alternative

The separate ELoFTR runner preserves the same timestamp handling, RANSAC,
quality gate, EKF, and evaluation outputs, but replaces only SuperPoint/LightGlue
matching. Install the dependencies first; its first run downloads the official
`zju-community/efficientloftr` checkpoint.

```bash
pip install -r requirements.txt

python scripts/run_elo_ftr_on_video.py \
  --video /Users/macbook/Documents/VO/data/videos/camera_down.mp4 \
  --timestamps /Users/macbook/Documents/VO/data/camera_down_index.csv \
  --heading-imu /Users/macbook/Documents/VO/data/imu.csv \
  --imu-heading-yaw-sign -1 --down-camera \
  --ground-truth /Users/macbook/Documents/VO/data/odometry.csv \
  --frame-stride 5 --min-parallax-px 18 \
  --tile-spacing-m 0.30 --tile-spacing-px 34 \
  --output /Users/macbook/Documents/VO/output/trajectory_05_down_elo_ftr
```

This model was pretrained for outdoor image matching, not underwater tile grids.
Run the ground-truth comparison before deciding it is an improvement.

### Evaluate the recorded down-camera trajectory

`trajectory_05` uses real, irregular camera timestamps. Do not use the MP4's
nominal frame rate for sensor fusion; pass the camera index CSV. The recording's
IMU quaternion is used only as an absolute heading source here. Raw accelerometer
integration is intentionally disabled until its frame convention and biases have
been calibrated.

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python scripts/run_vo_on_video.py \
  --video /Users/macbook/Documents/VO/data/videos/camera_down.mp4 \
  --timestamps /Users/macbook/Documents/VO/data/camera_down_index.csv \
  --heading-imu /Users/macbook/Documents/VO/data/imu.csv \
  --imu-heading-yaw-sign -1 \
  --down-camera \
  --ground-truth /Users/macbook/Documents/VO/data/odometry.csv \
  --frame-stride 5 \
  --min-parallax-px 18 \
  --tile-spacing-m 0.30 --tile-spacing-px 34 \
  --output /Users/macbook/Documents/VO/output/trajectory_05_down_optimized
```

This writes `trajectory_comparison.png`, `trajectory_metrics.json`, and a
timestamped `trajectory_comparison.csv`. The comparison uses SE(2) alignment to
remove the arbitrary initial XY origin and heading. It is a diagnostic, not a
claim of absolute navigation accuracy.

`--min-parallax-px 18` was tuned on the supplied `trajectory_05` recording;
re-evaluate this threshold on a separate recording before using it elsewhere.

### IMU CSV format

```
timestamp,ax,ay,az,gx,gy,gz,roll,pitch,yaw
```

Units: accel m/s², gyro rad/s, angles rad. See `data/imu_sample.csv`.

### Pressure CSV format

```
timestamp,pressure_pa
```

See `data/pressure_sample.csv`. Calibrate `p_surface_pa` in `config/default.yaml` at pool surface before each run.

## Configuration

Edit `config/default.yaml`:

- **`tiles.spacing_m`** — measure center-to-center grout distance in your pool (typically 0.25–0.35 m)
- **`camera.fx/fy/cx/cy`** — from checkerboard calibration
- **`quality_gate.*`** — tighten if you get false visual accepts
- **`ekf.sigma_*`** — tune after logging IMU vs visual disagreement

## SAUVC mission mapping

| Task | This pipeline | You still need |
|------|---------------|----------------|
| Qualification / Navigation | Dead-reckoning + VO for ~16 m gate approach | Gate detector (color stripes / shape) |
| Target Acquisition | Position estimate to search pattern | Blue/red drum classifier |
| Communication | Waypoints from fused state | Red/yellow/blue flare detector |
| Depth safety | Pressure + EKF Z | Altitude controller (PID on depth) |

## Recommended next steps

1. Measure tile spacing and calibrate camera in the practice pool.
2. Log synchronized IMU + camera + pressure on all 5 videos.
3. Tune `quality_gate.min_inlier_ratio` until bad frames are rejected.
4. Add a **grid-line detector** as backup when LightGlue fails (underwater blur).
5. Build task-specific detectors (gate, drums, flares) on top of this state estimate.

## Project layout

```
config/default.yaml       # tuning parameters
src/features/             # SuperPoint/LightGlue + quality gate
src/motion/               # RANSAC homography + tile scale
src/fusion/               # EKF + sensor loaders
src/pipeline/             # end-to-end VO pipeline
scripts/                  # CLI entry points
data/                     # sample frame + CSV templates
```
