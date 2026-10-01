from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Cache timestamp vectors keyed by (id, length) of sensor lists.
_TIME_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _sample_times(samples: list) -> np.ndarray:
    key = (id(samples), len(samples))
    cached = _TIME_CACHE.get(key)
    if cached is not None:
        return cached
    times = np.fromiter((s.timestamp for s in samples), dtype=float, count=len(samples))
    _TIME_CACHE[key] = times
    return times


@dataclass
class IMUSample:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None


@dataclass
class PressureSample:
    timestamp: float
    pressure_pa: float


def load_imu_csv(path: str) -> list[IMUSample]:
    df = pd.read_csv(path)
    # Accept both the project's original schema and sauvc_traj_recorder's schema.
    # The recorder stores ROS angular velocity as wx/wy/wz and time as t.
    df = df.rename(columns={
        "t": "timestamp",
        "wx": "gx",
        "wy": "gy",
        "wz": "gz",
    })
    required = {"timestamp", "ax", "ay", "az", "gx", "gy", "gz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IMU CSV missing columns: {missing}")

    # Recorder IMU messages contain an orientation quaternion.  Convert it once
    # here and unwrap it so interpolation remains continuous through +/- pi.
    if {"qx", "qy", "qz", "qw"}.issubset(df.columns) and "yaw" not in df.columns:
        qx = df["qx"].to_numpy(dtype=float)
        qy = df["qy"].to_numpy(dtype=float)
        qz = df["qz"].to_numpy(dtype=float)
        qw = df["qw"].to_numpy(dtype=float)
        df["yaw"] = np.unwrap(np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        ))

    samples = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        samples.append(
            IMUSample(
                timestamp=float(d["timestamp"]),
                ax=float(d["ax"]),
                ay=float(d["ay"]),
                az=float(d["az"]),
                gx=float(d["gx"]),
                gy=float(d["gy"]),
                gz=float(d["gz"]),
                roll=float(d["roll"]) if "roll" in d else None,
                pitch=float(d["pitch"]) if "pitch" in d else None,
                yaw=float(d["yaw"]) if "yaw" in d else None,
            )
        )
    return sorted(samples, key=lambda sample: sample.timestamp)


def load_pressure_csv(path: str) -> list[PressureSample]:
    df = pd.read_csv(path)
    # sauvc_traj_recorder uses ROS's field name, fluid_pressure.
    df = df.rename(columns={"t": "timestamp", "fluid_pressure": "pressure_pa"})
    if "timestamp" not in df.columns or "pressure_pa" not in df.columns:
        raise ValueError("Pressure CSV needs columns: timestamp, pressure_pa")
    return sorted([
        PressureSample(timestamp=float(r.timestamp), pressure_pa=float(r.pressure_pa))
        for r in df.itertuples(index=False)
    ], key=lambda sample: sample.timestamp)


def pressure_to_depth(
    pressure_pa: float,
    p_surface_pa: float = 101325.0,
    rho: float = 998.0,
    g: float = 9.81,
) -> float:
    """Depth below surface in meters (positive down)."""
    return max(0.0, (pressure_pa - p_surface_pa) / (rho * g))


def interpolate_imu(samples: list[IMUSample], t: float) -> IMUSample | None:
    if not samples:
        return None
    if t <= samples[0].timestamp:
        return samples[0]
    if t >= samples[-1].timestamp:
        return samples[-1]

    times = _sample_times(samples)
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = max(0, min(i, len(samples) - 2))
    a, b = samples[i], samples[i + 1]
    denom = b.timestamp - a.timestamp
    alpha = 0.0 if denom <= 0 else (t - a.timestamp) / denom
    return IMUSample(
        timestamp=t,
        ax=a.ax + alpha * (b.ax - a.ax),
        ay=a.ay + alpha * (b.ay - a.ay),
        az=a.az + alpha * (b.az - a.az),
        gx=a.gx + alpha * (b.gx - a.gx),
        gy=a.gy + alpha * (b.gy - a.gy),
        gz=a.gz + alpha * (b.gz - a.gz),
        roll=_lerp(a.roll, b.roll, alpha),
        pitch=_lerp(a.pitch, b.pitch, alpha),
        yaw=_lerp(a.yaw, b.yaw, alpha),
    )


def interpolate_depth(samples: list[PressureSample], t: float) -> float | None:
    if not samples:
        return None
    if t <= samples[0].timestamp:
        return pressure_to_depth(samples[0].pressure_pa)
    if t >= samples[-1].timestamp:
        return pressure_to_depth(samples[-1].pressure_pa)

    times = _sample_times(samples)
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = max(0, min(i, len(samples) - 2))
    a, b = samples[i], samples[i + 1]
    denom = b.timestamp - a.timestamp
    alpha = 0.0 if denom <= 0 else (t - a.timestamp) / denom
    p = a.pressure_pa + alpha * (b.pressure_pa - a.pressure_pa)
    return pressure_to_depth(p)


def _lerp(a: float | None, b: float | None, alpha: float) -> float | None:
    if a is None or b is None:
        return a if a is not None else b
    return a + alpha * (b - a)
