import torch.nn as nn

from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN


class RevINLinear(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {}

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.fc = nn.Linear(input_len, output_len)
        self.revin = RevIN(self.n_features, affine=True)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # normalize
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.fc(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_feature)
        x = self.revin(x, mode="denorm")  # denormalize back
        return x
