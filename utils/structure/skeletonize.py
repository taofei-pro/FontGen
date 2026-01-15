from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize as sk_skeletonize


def build_skeleton_map(image: np.ndarray, threshold: float = 0.5) -> torch.Tensor:
    """Generate a skeleton map from a [0,1] normalized image."""
    binary = image >= threshold
    skeleton = sk_skeletonize(binary).astype(np.float32)
    return torch.from_numpy(skeleton).unsqueeze(0)
