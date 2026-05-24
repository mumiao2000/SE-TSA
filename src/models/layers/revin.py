import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")
        return x

    def _get_statistics(self, x: torch.Tensor):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * 1e-5)
        x = x * self.stdev
        x = x + self.mean
        return x
