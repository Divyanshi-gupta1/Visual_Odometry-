"""VO pipeline variant that replaces SuperPoint/LightGlue with EfficientLoFTR."""

from __future__ import annotations

from src.features.efficient_loftr import EfficientLoFTRMatcher
from src.features.superpoint_lightglue import FeatureMatcher
from src.fusion.ekf import AUVExtendedKalmanFilter
from src.pipeline.vo_pipeline import VOPipeline
from src.utils.io import CameraIntrinsics


class ELoFTRVOPipeline(VOPipeline):
    """Same timestamp, EKF, RANSAC, and quality-gate flow; E-LoFTR matches only."""

    def __init__(self, config: dict, camera: CameraIntrinsics) -> None:
        # Do not call VOPipeline.__init__: it would construct SuperPoint/ORB first.
        self.config = config
        self.camera = camera
        matcher_cfg = config.get("elo_ftr", {})
        self.matcher = EfficientLoFTRMatcher(
            model_id=str(matcher_cfg.get("model_id", "zju-community/efficientloftr")),
            confidence_threshold=float(matcher_cfg.get("confidence_threshold", 0.2)),
            device=matcher_cfg.get("device"),
            max_side=int(matcher_cfg.get("max_side", 640)),
        )
        self.ekf = AUVExtendedKalmanFilter(config)
        features = config.get("features", {})
        self.quality_cfg = {**features, **config.get("quality_gate", {})}
        self.tile_spacing = float(config.get("tiles", {}).get("spacing_m", 0.30))
        self.tile_spacing_px = float(config.get("tiles", {}).get("spacing_px", 0))
        self.max_step_m = float(matcher_cfg.get("max_step_m", 0.0))
        if self.max_step_m <= 0:
            self.max_step_m = float(config.get("visual", {}).get("max_step_m", 0.80))
        # LK fallback fights LoFTR on this repetitive tile floor; keep E-LoFTR only.
        self.fallback_matcher = None
