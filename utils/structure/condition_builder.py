from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from configs.structure_config import StructureConfig
from utils.structure.edge_map import build_edge_map
from utils.structure.mask_generator import build_component_mask
from utils.structure.skeletonize import build_skeleton_map


@dataclass
class StructureCondition:
    tensor: torch.Tensor
    meta: dict[str, Any]


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image to [0, 1] for structure extraction."""
    if image.dtype != torch.float32:
        image = image.float()
    return (image * 0.5 + 0.5).clamp(0.0, 1.0)


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 3:
        image = image.squeeze(0)
    return image.detach().cpu().numpy()


def build_structure_condition(
    sample: dict[str, Any],
    config: StructureConfig,
) -> StructureCondition:
    """Build structure condition maps from a dataset sample."""
    image = sample["tgt_img"]
    image_norm = _normalize_image(image)
    image_np = _to_numpy(image_norm)

    condition_maps: list[torch.Tensor] = []
    meta: dict[str, Any] = {}

    if config.use_component_mask:
        mask = build_component_mask(image_np)
        condition_maps.append(mask)
        meta["component_mask"] = True

    if config.use_edge_map:
        edge = build_edge_map(image_np)
        condition_maps.append(edge)
        meta["edge_map"] = True

    if config.use_skeleton:
        skeleton = build_skeleton_map(image_np)
        condition_maps.append(skeleton)
        meta["skeleton_map"] = True

    if not condition_maps:
        raise ValueError("No structure condition enabled in StructureConfig.")

    condition_tensor = torch.cat(condition_maps, dim=0).to(image.device)
    return StructureCondition(tensor=condition_tensor, meta=meta)
