#!/usr/bin/env python3
"""Run the normal AUV VO pipeline with EfficientLoFTR instead of SuperPoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.argv = [str(ROOT / "scripts" / "run_vo_on_video.py"), "--matcher", "elo_ftr", *sys.argv[1:]]
runpy.run_path(str(ROOT / "scripts" / "run_vo_on_video.py"), run_name="__main__")
