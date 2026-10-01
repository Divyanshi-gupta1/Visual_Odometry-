from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    kpts0: np.ndarray  # (N, 2)
    kpts1: np.ndarray  # (N, 2)
    scores: np.ndarray  # (N,)
    desc0: np.ndarray | None = None
    desc1: np.ndarray | None = None


class FeatureMatcher:
    """SuperPoint + LightGlue with tile-corner and ORB fallbacks."""

    def __init__(
        self,
        backend: str = "auto",
        max_keypoints: int = 512,
        keypoint_threshold: float = 0.005,
        match_threshold: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.backend = backend
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.match_threshold = match_threshold
        self._device = device
        self._extractor = None
        self._matcher = None
        self._orb = None
        self._bf = None
        self._prev_corners: np.ndarray | None = None
        self._init_backend()

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _init_backend(self) -> None:
        use_lightglue = self.backend in ("auto", "lightglue")
        if use_lightglue:
            try:
                import torch
                from lightglue import LightGlue, SuperPoint

                device = self._resolve_device()
                self._extractor = SuperPoint(
                    max_num_keypoints=self.max_keypoints,
                    detection_threshold=self.keypoint_threshold,
                ).eval().to(device)
                self._matcher = LightGlue(
                    features="superpoint",
                    depth_confidence=-1,
                    width_confidence=-1,
                    filter_threshold=self.match_threshold,
                ).eval().to(device)
                self.backend = "lightglue"
                self._device = device
                return
            except Exception:
                if self.backend == "lightglue":
                    raise

        # Underwater tile floors: ORB often fails; prefer tile-corner tracker
        if self.backend in ("auto", "tile_corners"):
            self.backend = "tile_corners"
            return

        self.backend = "orb"
        self._orb = cv2.ORB_create(nfeatures=self.max_keypoints)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Underwater images: boost local contrast for grout/tile edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _preprocess_for_deep(img_gray: np.ndarray) -> "torch.Tensor":
        import torch

        t = torch.from_numpy(img_gray).float() / 255.0
        return t[None, None]

    def extract_and_match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        g0 = self._to_gray(img0)
        g1 = self._to_gray(img1)

        if self.backend == "lightglue":
            return self._match_lightglue(g0, g1)
        if self.backend == "tile_corners":
            return self._match_tile_corners(g0, g1)
        return self._match_orb(g0, g1)

    def _detect_tile_corners(self, gray: np.ndarray) -> np.ndarray:
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_keypoints,
            qualityLevel=0.02,
            minDistance=12,
            blockSize=7,
            useHarrisDetector=True,
            k=0.04,
        )
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def _match_tile_corners(self, g0: np.ndarray, g1: np.ndarray) -> MatchResult:
        pts0 = self._detect_tile_corners(g0)
        if len(pts0) < 8:
            return MatchResult(
                kpts0=np.empty((0, 2), dtype=np.float32),
                kpts1=np.empty((0, 2), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        p0 = pts0.reshape(-1, 1, 2)
        p1, status, err = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
        status = status.ravel().astype(bool)
        err = err.ravel()

        k0 = pts0[status]
        k1 = p1.reshape(-1, 2)[status]
        scores = 1.0 / (1.0 + err[status])

        return MatchResult(kpts0=k0, kpts1=k1, scores=scores)

    def _match_lightglue(self, g0: np.ndarray, g1: np.ndarray) -> MatchResult:
        import torch

        device = self._device or "cpu"
        t0 = self._preprocess_for_deep(g0).to(device)
        t1 = self._preprocess_for_deep(g1).to(device)

        with torch.inference_mode():
            feats0 = self._extractor.extract(t0)
            feats1 = self._extractor.extract(t1)
            matches = self._matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches = [
                {k: v.cpu() for k, v in x.items()} for x in [feats0, feats1, matches]
            ]

        kpts0 = feats0["keypoints"][0].numpy()
        kpts1 = feats1["keypoints"][0].numpy()
        m_idx = matches["matches"][0].numpy()
        scores = matches["scores"][0].numpy()

        valid = m_idx[:, 0] >= 0
        idx0 = m_idx[valid, 0]
        idx1 = m_idx[valid, 1]
        return MatchResult(
            kpts0=kpts0[idx0],
            kpts1=kpts1[idx1],
            scores=scores[valid],
        )

    def _match_orb(self, g0: np.ndarray, g1: np.ndarray) -> MatchResult:
        self._orb = cv2.ORB_create(nfeatures=self.max_keypoints)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        k0, d0 = self._orb.detectAndCompute(g0, None)
        k1, d1 = self._orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return self._match_tile_corners(g0, g1)

        raw = self._bf.knnMatch(d0, d1, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 8:
            return self._match_tile_corners(g0, g1)

        pts0 = np.float32([k0[m.queryIdx].pt for m in good])
        pts1 = np.float32([k1[m.trainIdx].pt for m in good])
        scores = np.float32([1.0 - m.distance / 256.0 for m in good])
        return MatchResult(kpts0=pts0, kpts1=pts1, scores=scores)
