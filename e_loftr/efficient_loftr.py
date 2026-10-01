"""EfficientLoFTR (E-LoFTR) matcher for the AUV visual-odometry pipeline.

Tries the official Hugging Face checkpoint first (`zju-community/efficientloftr`).
If that stack is unavailable, falls back to Kornia's detector-free LoFTR, which
is the same family of dense matchers and keeps the MatchResult contract unchanged.
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np

from src.features.superpoint_lightglue import MatchResult


class EfficientLoFTRMatcher:
    """Detector-free E-LoFTR / LoFTR matcher with explicit dependency checks."""

    def __init__(
        self,
        model_id: str = "zju-community/efficientloftr",
        confidence_threshold: float = 0.2,
        device: str | None = None,
        max_side: int = 640,
    ) -> None:
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.max_side = int(max_side)
        self.backend = "none"
        self._processor = None
        self._model = None
        self._torch = None
        self._load_model()

    def _pick_device(self, torch_mod) -> str:
        if self.device:
            return self.device
        if torch_mod.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "EfficientLoFTR requires torch. Install with: pip install torch torchvision"
            )

        import torch

        self._torch = torch
        self.device = self._pick_device(torch)

        hf_ok = importlib.util.find_spec("transformers") is not None
        if hf_ok:
            try:
                from transformers import AutoImageProcessor, AutoModelForKeypointMatching

                self._processor = AutoImageProcessor.from_pretrained(self.model_id)
                self._model = (
                    AutoModelForKeypointMatching.from_pretrained(self.model_id)
                    .eval()
                    .to(self.device)
                )
                self.backend = "efficientloftr"
                return
            except Exception:
                try:
                    from transformers import AutoImageProcessor, AutoModel

                    self._processor = AutoImageProcessor.from_pretrained(self.model_id)
                    self._model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)
                    self.backend = "efficientloftr"
                    return
                except Exception:
                    self._processor = None
                    self._model = None

        if importlib.util.find_spec("kornia") is None:
            raise RuntimeError(
                "Could not load Hugging Face EfficientLoFTR and kornia is not installed. "
                "Fix torch/transformers or: pip install kornia"
            )

        import kornia.feature as KF

        pretrained = "indoor_new" if "indoor" in self.model_id.lower() else "outdoor"
        self._model = KF.LoFTR(pretrained=pretrained).eval().to(self.device)
        self.backend = "kornia_loftr"

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def _resize(self, gray: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = gray.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale >= 0.999:
            return gray, 1.0, 1.0
        new_w = int(round(w * scale / 8.0) * 8)
        new_h = int(round(h * scale / 8.0) * 8)
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, w / new_w, h / new_h

    def extract_and_match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        gray0, gray1 = self._gray(img0), self._gray(img1)
        if self.backend == "efficientloftr":
            return self._match_hf(gray0, gray1)
        return self._match_kornia(gray0, gray1)

    def _match_hf(self, gray0: np.ndarray, gray1: np.ndarray) -> MatchResult:
        from PIL import Image

        images = [Image.fromarray(gray0), Image.fromarray(gray1)]
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        matches = self._processor.post_process_keypoint_matching(
            outputs,
            [[(gray0.shape[0], gray0.shape[1]), (gray1.shape[0], gray1.shape[1])]],
            threshold=self.confidence_threshold,
        )[0]
        kpts0 = matches["keypoints0"].detach().cpu().numpy().astype(np.float32)
        kpts1 = matches["keypoints1"].detach().cpu().numpy().astype(np.float32)
        scores = matches["matching_scores"].detach().cpu().numpy().astype(np.float32)
        return MatchResult(kpts0=kpts0, kpts1=kpts1, scores=scores)

    def _match_kornia(self, gray0: np.ndarray, gray1: np.ndarray) -> MatchResult:
        torch = self._torch
        r0, sx0, sy0 = self._resize(gray0)
        r1, sx1, sy1 = self._resize(gray1)
        t0 = torch.from_numpy(r0).float().div(255.0)[None, None].to(self.device)
        t1 = torch.from_numpy(r1).float().div(255.0)[None, None].to(self.device)
        with torch.inference_mode():
            out = self._model({"image0": t0, "image1": t1})
        kpts0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
        kpts1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
        scores = out["confidence"].detach().cpu().numpy().astype(np.float32)
        keep = scores >= self.confidence_threshold
        kpts0 = kpts0[keep]
        kpts1 = kpts1[keep]
        scores = scores[keep]
        if len(kpts0):
            kpts0[:, 0] *= sx0
            kpts0[:, 1] *= sy0
            kpts1[:, 0] *= sx1
            kpts1[:, 1] *= sy1
        return MatchResult(kpts0=kpts0, kpts1=kpts1, scores=scores)
