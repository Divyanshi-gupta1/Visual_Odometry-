#!/usr/bin/env python3
"""
Demo feature detection + motion on sample tile frame(s).

If only one image is provided, applies a small synthetic shift so you can
verify the pipeline before your full 5 videos are wired in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.vo_pipeline import VOPipeline
from src.utils.io import camera_from_config, ensure_dir, load_config
from src.utils.visualization import annotate_frame, draw_matches


def synthetic_shift(img: np.ndarray, dx: int = 8, dy: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo VO on 1-2 tile images")
    parser.add_argument(
        "--image0",
        default="data/sample_frame.png",
        help="First frame (or only frame)",
    )
    parser.add_argument("--image1", default=None, help="Second frame (optional)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="output")
    parser.add_argument("--synthetic-shift", action="store_true",
                        help="Shift image0 to create image1 when only one frame exists")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    camera = camera_from_config(cfg)
    out_dir = ensure_dir(ROOT / args.output)

    img0 = cv2.imread(str(ROOT / args.image0))
    if img0 is None:
        raise FileNotFoundError(f"Cannot read {args.image0}")

    if args.image1:
        img1 = cv2.imread(str(ROOT / args.image1))
        if img1 is None:
            raise FileNotFoundError(f"Cannot read {args.image1}")
    else:
        img1 = synthetic_shift(img0)
        print("Note: only one frame provided — using synthetic shift for demo.")

    pipeline = VOPipeline(cfg, camera)
    motion, matches = pipeline.process_image_pair(img0, img1)

    inliers = motion.inlier_mask if motion.inlier_mask is not None else None
    vis = draw_matches(img0, img1, matches.kpts0, matches.kpts1, inliers)
    cv2.imwrite(str(out_dir / "matches.png"), vis)

    lines = [
        f"backend: {pipeline.matcher.backend}",
        f"matches: {len(matches.kpts0)}",
        f"accepted: {motion.accepted} ({motion.reason})",
    ]
    if motion.quality:
        q = motion.quality.metrics
        lines += [
            f"inliers: {q.num_inliers}/{q.num_matches} ({q.inlier_ratio:.2f})",
            f"reproj err: {q.mean_reproj_error:.2f} px",
            f"parallax: {q.median_parallax:.1f} px",
        ]
    if motion.accepted:
        lines += [
            f"tx: {motion.tx*100:.1f} cm",
            f"ty: {motion.ty*100:.1f} cm",
            f"dyaw: {np.degrees(motion.dyaw):.2f} deg",
        ]

    annotated = annotate_frame(img0, lines)
    cv2.imwrite(str(out_dir / "summary.png"), annotated)

    print("\n".join(lines))
    print(f"Saved: {out_dir / 'matches.png'}")
    print(f"Saved: {out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
