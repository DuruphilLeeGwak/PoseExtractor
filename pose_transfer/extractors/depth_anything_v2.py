"""Depth Anything V2 extractor (optional).

This module is optional. If the dependency is missing, it will raise at runtime.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import os
import sys
import urllib.request

import cv2
import numpy as np


class DepthAnythingV2Extractor:
    def __init__(self, model: str = "depth_anything_v2_vitl", device: str = "cuda", repo_path: Optional[str] = None):
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for DepthAnythingV2Extractor") from exc

        self.torch = torch
        self.device = device
        self.model_name = model

        self.repo_path = repo_path

        try:
            if repo_path:
                repo_dir = Path(repo_path)
                if not repo_dir.exists():
                    raise FileNotFoundError(f"Depth Anything repo not found: {repo_dir}")

                # Add repo to sys.path for local import
                if str(repo_dir) not in sys.path:
                    sys.path.insert(0, str(repo_dir))

                from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }

                # Normalize model name
                encoder = model.replace("depth_anything_v2_", "")
                if encoder not in model_configs:
                    encoder = "vitl"

                checkpoints_dir = repo_dir / "checkpoints"
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                ckpt_name = f"depth_anything_v2_{encoder}.pth"
                ckpt_path = checkpoints_dir / ckpt_name

                if not ckpt_path.exists():
                    url_map = {
                        'vits': "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
                        'vitb': "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
                        'vitl': "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
                    }
                    if encoder not in url_map:
                        raise RuntimeError(f"No checkpoint URL for encoder: {encoder}")
                    urllib.request.urlretrieve(url_map[encoder], str(ckpt_path))

                self.model = DepthAnythingV2(**model_configs[encoder])
                self.model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu'))
            else:
                # Fallback to torch.hub (requires internet + hubconf)
                self.model = torch.hub.load(
                    "DepthAnything/Depth-Anything-V2",
                    model,
                    pretrained=True,
                    trust_repo=True
                )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to load Depth Anything V2. Provide a valid local repo_path "
                "or ensure internet access for torch.hub."
            ) from exc

        self.model.to(device).eval()

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth map for an image (BGR). Returns HxW float32 depth normalized to [0,1]."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        depth = None
        # Try model-specific inference helper if available
        if hasattr(self.model, "infer_image"):
            depth = self.model.infer_image(img_rgb)
        else:
            img = img_rgb.astype(np.float32) / 255.0
            img = self.torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                pred = self.model(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            depth = pred.squeeze().detach().cpu().numpy()

        if depth is None:
            raise RuntimeError("DepthAnythingV2Extractor failed to produce depth output")

        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        depth = depth.astype(np.float32)
        d_min = np.percentile(depth, 1)
        d_max = np.percentile(depth, 99)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        depth = np.clip(depth, 0.0, 1.0)
        return depth
