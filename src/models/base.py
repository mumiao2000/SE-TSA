"""Base forecast model interface."""

from typing import Dict

import torch
import torch.nn as nn


class BaseForecastModel(nn.Module):
    def __init__(
        self, input_len: int, output_len: int, n_features: int, config: Dict = None
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.config = config if config is not None else self.get_default_config()

    @classmethod
    def get_all_configs(cls) -> Dict[str, list]:
        return {}

    @classmethod
    def get_default_config(cls) -> Dict[str, any]:
        all_configs = cls.get_all_configs()
        return {k: v[len(v) // 2] for k, v in all_configs.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
