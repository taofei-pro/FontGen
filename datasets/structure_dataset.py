from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from configs.structure_config import StructureConfig
from utils.structure.condition_builder import build_structure_condition


class StructureConditionDataset(Dataset):
    """Dataset wrapper that returns images with structure condition maps."""

    def __init__(
        self,
        base_dataset: Dataset,
        config: StructureConfig | None = None,
        condition_builder: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.config = config or StructureConfig()
        self.condition_builder = condition_builder

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base_dataset[idx]
        image = sample["tgt_img"] if isinstance(sample, dict) else sample
        if self.condition_builder is None:
            condition = build_structure_condition(sample, self.config)
        else:
            condition = self.condition_builder(sample)
        meta = {"index": idx}
        return {
            "image": image,
            "condition": condition.tensor,
            "meta": {**meta, **condition.meta},
        }
