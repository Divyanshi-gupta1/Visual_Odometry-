from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def camera_from_config(cfg: dict[str, Any]) -> CameraIntrinsics:
    c = cfg["camera"]
    return CameraIntrinsics(
        fx=float(c["fx"]),
        fy=float(c["fy"]),
        cx=float(c["cx"]),
        cy=float(c["cy"]),
        width=int(c["width"]),
        height=int(c["height"]),
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
