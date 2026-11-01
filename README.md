# AUV Visual Odometry & Multi-Sensor Fusion Pipeline

An end-to-end Visual Odometry (VO) and sensor fusion system developed for down-facing underwater camera navigation on an Autonomous Underwater Vehicle (AUV). 

The system fuses visual motion estimates with 6-DoF IMU orientation and hydrostatic pressure depth in an Extended Kalman Filter (EKF), achieving **sub-meter trajectory accuracy (< 0.79 m max peak error)** over a full 8-minute underwater mission.

---

## Key Highlights & Features

- **Dual Matching Backends:**
  - **SuperPoint + LightGlue:** Keypoint detector and attentional matcher with RANSAC homography and metric scale recovery from pool floor tile spacing (0.30 m).
  - **EfficientLoFTR (E-LoFTR):** Detector-free Transformer matcher that handles repetitive, low-texture underwater pool tiles by generating 700+ dense correspondences per pair.
- **Sensor Fusion (7-State EKF):**
  - Fusion of visual displacement with IMU 3D attitude ($R_{wb}$) and hydrostatic depth.
  - Coordinate frame transformations bridging IMU (ENU/FLU) to navigation frame (NED/FRD).
  - Hydrostatic gauge pressure calibration achieving **0.87 cm depth RMSE**.
  - Off-center camera lever-arm compensation for rotational yaw artifacts.
- **Evaluation & Diagnostics:**
  - Evaluated against ground truth odometry with SE(2) trajectory alignment.
  - Low-frequency pose-graph drift relaxation reducing mission drift by over 80%.
  - Configurable YAML quality gates (reprojection error, inlier ratio, parallax, rotation limits).
  - Batch runner logging per-frame acceptance rate, inlier ratio, and tracking residuals.

---

## System Architecture

```
                          Down-Facing Camera (MP4)
                                    │
                         [Every Nth Frame Stride]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [SuperPoint + LightGlue]         [EfficientLoFTR Transformer]
     (Keypoint & Descriptor Match)        (Dense Global Cross-Attention)
                    │                               │
       RANSAC Homography Estimation     RANSAC 2D Affine Flow Estimation
       + Tile Grid (0.30 m) Scale       + Metric Altitude Scaling
                    │                               │
       Quality Gate Filtering           Lever-Arm Rotational Compensation
                    └───────────────┬───────────────┘
                                    │ Body Displacement (dx, dy)
                                    ▼
IMU (qx, qy, qz, qw, wz) ──► [Extended Kalman Filter] ◄── Pressure Sensor (Pa)
 (3D Attitude & Heading)       (7-State: x,y,z,vx,vy,vz,yaw)   (Hydrostatic Depth)
                                    │
                             Raw Trajectory
                                    │
                     [Pose-Graph Drift Relaxation]
                                    │
                                    ▼
                      Evaluated Metric Trajectory
```

---

## Performance Benchmark

Evaluated on the full 544-second mission (`camera_down.mp4`, 3,342 frames, 473 s active navigation):

| Metric | SuperPoint + LightGlue Baseline | EfficientLoFTR + EKF (This Work) |
| :--- | :--- | :--- |
| **Feature / Match Philosophy** | Sparse detector + descriptor | Detector-free dense Transformer |
| **Matches per Frame Pair** | 30 – 80 (repetitive tile noise) | **700 – 1,200+ dense matches** |
| **Planar RMSE (ATE)** | ~2.27 m | **0.382 m (38.2 cm)** |
| **Planar Max Peak Error** | > 4.5 m | **0.785 m (78.5 cm)** |
| **X-Axis (North) RMSE** | ~2.10 m | **0.348 m (34.8 cm)** |
| **Y-Axis (East) RMSE** | ~0.85 m | **0.156 m (15.6 cm)** |
| **Depth (Z) RMSE** | ~0.74 m (raw) | **0.0087 m (0.87 cm)** |

---

## Project Structure

```
├── config/
│   └── default.yaml              # Camera intrinsics, noise covariances, thresholds
├── data/
│   ├── videos/
│   │   └── camera_down.mp4       # Down-facing underwater camera footage
│   ├── camera_down_index.csv     # Exact video frame timestamps
│   ├── imu.csv                   # IMU orientation quaternions and angular rates
│   ├── pressure.csv              # Fluid pressure depth sensor readings
│   └── odometry.csv              # Ground truth odometry for benchmarking
├── e_loftr/
│   ├── efficient_loftr.py        # Detector-free E-LoFTR / LoFTR matcher wrapper
│   ├── elo_ftr_motion.py         # Affine motion estimation & lever-arm compensation
│   ├── elo_ftr_ekf.py            # 7-state EKF fusing visual, IMU, and depth
│   ├── elo_ftr_pipeline.py       # Orchestrator synchronizing video, IMU, and pressure
│   ├── elo_ftr_vo_pipeline.py    # E-LoFTR matcher plugin for baseline batch runner
│   └── run_elo_ftr.py            # Main CLI runner with relaxation and decomposition plots
├── src/
│   ├── features/
│   │   ├── superpoint_lightglue.py # SuperPoint extractor and LightGlue matcher
│   │   └── quality_gate.py       # Acceptance filters (inliers, parallax, reprojection)
│   ├── fusion/
│   │   ├── ekf.py                # Baseline AUV EKF implementation
│   │   └── sensors.py            # Sensor loading and synchronization helpers
│   ├── motion/
│   │   └── ransac_motion.py      # RANSAC homography & tile-scale motion estimator
│   ├── pipeline/
│   │   └── vo_pipeline.py        # Baseline visual odometry pipeline
│   └── utils/
│       ├── io.py                 # YAML config loader and camera dataclasses
│       ├── trajectory_eval.py    # SE(2) trajectory alignment and ATE computation
│       └── visualization.py      # Matplotlib trajectory and error plotting
├── scripts/
│   ├── run_vo_on_video.py        # Configurable batch runner with diagnostic logging
│   └── demo_on_sample.py         # Single-frame pair demonstration script
├── tests/
│   └── test_ekf.py               # Unit test suite for EKF implementations
├── output/
│   └── elo_ftr_full/             # Benchmark plots, CSV comparison, and metrics JSON
└── requirements.txt              # Project dependencies
```

---

## Installation & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Divyanshi-gupta1/Visual_Odometry-.git
cd Visual_Odometry-

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Optional: Install LightGlue (for SuperPoint baseline)
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

---

## How to Run

### 1. Run the High-Accuracy E-LoFTR Pipeline (Recommended)
Runs the detector-free Transformer pipeline with calibrated affine motion, EKF sensor fusion, and pose-graph relaxation:

```bash
python e_loftr/run_elo_ftr.py \
    --video data/videos/camera_down.mp4 \
    --start-frame 508 \
    --frame-stride 8 \
    --output output/elo_ftr_full
```

**Outputs generated in `--output`:**
- `trajectory_comparison.png` — Top-down trajectory overlaid against Ground Truth.
- `error_decomposition.png` — 3-panel error plots (Absolute Planar, X-Axis North, Y-Axis East).
- `trajectory.png` — Standalone estimated path.
- `trajectory_metrics.json` — Comprehensive numeric evaluation metrics.
- `trajectory_comparison.csv` — Synchronized per-frame estimated vs. ground truth positions and errors.

### 2. Run the SuperPoint Baseline Batch Runner
Runs the baseline SuperPoint/LightGlue pipeline with quality gating and tile-grid scaling:

```bash
python scripts/run_vo_on_video.py \
    --video data/videos/camera_down.mp4 \
    --timestamps data/camera_down_index.csv \
    --heading-imu data/imu.csv \
    --imu-heading-yaw-sign -1 \
    --down-camera \
    --ground-truth data/odometry.csv \
    --frame-stride 5 \
    --min-parallax-px 18 \
    --tile-spacing-m 0.30 \
    --tile-spacing-px 34 \
    --output output/dataset_run
```

---

## Sensor Calibration & Engineering Notes

### 1. Hydrostatic Depth & Gauge Pressure
The pressure sensor output (`pressure.csv`) records **gauge pressure** ($\approx 5.7\text{ kPa}$ at depth), not absolute atmospheric pressure ($101.3\text{ kPa}$). Metric depth is calculated via:
$$z = \frac{p_{\text{gauge}}}{\rho_{\text{water}} \cdot g} + z_{\text{offset}} = \frac{p_{\text{gauge}}}{998.0 \cdot 9.81} + 0.09\text{ m}$$
Using this formula avoids negative pressure clamping, lowering depth RMSE from **74 cm down to 0.87 cm**.

### 2. Off-Center Camera Lever-Arm Correction
The downward camera is positioned with an offset of $\approx 50.8\text{ cm}$ behind the vehicle's center of rotation (`LEVER_Y = -0.508`). When the vehicle yaws, the camera physically sweeps through an arc. The motion model explicitly compensates for this rotational velocity:
$$\Delta x_{\text{body}} = A_{00} t_{xn} h + A_{10} t_{yn} h - \text{LEVER}_Y \cdot \omega_z \Delta t$$
$$\Delta y_{\text{body}} = A_{01} t_{xn} h + A_{11} t_{yn} h + \text{LEVER}_X \cdot \omega_z \Delta t$$

### 3. Mission Start Frame (`--start-frame 508`)
The recording begins with an 83-second surface hover phase before active thruster translation starts. Starting evaluation at frame 508 isolates active navigation from stationary water surface jitter.
