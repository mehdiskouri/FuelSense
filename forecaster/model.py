"""Demand forecasting Temporal Convolutional Network models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalConv1d(nn.Module):
    """Conv1d with causal padding so outputs depend only on current and past timesteps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    """Residual TCN block with two causal convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual: nn.Module
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class DemandTCN(nn.Module):
    """TCN forecaster that emits 14-day quantile predictions [p10, p50, p90]."""

    N_FEATURES = 6
    LOOKBACK = 90
    HORIZON = 14
    N_QUANTILES = 3

    def __init__(self, hidden_channels: int = 32, kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            TCNBlock(self.N_FEATURES, hidden_channels, kernel_size, dilation=1, dropout=dropout),
            TCNBlock(hidden_channels, hidden_channels, kernel_size, dilation=2, dropout=dropout),
            TCNBlock(hidden_channels, hidden_channels, kernel_size, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.HORIZON * self.N_QUANTILES),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [batch, lookback, features] -> [batch, features, lookback]
        x = x.permute(0, 2, 1)
        x = self.backbone(x)
        last_timestep = x[:, :, -1]
        out = self.head(last_timestep)
        return out.view(-1, self.HORIZON, self.N_QUANTILES)


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression outputs."""

    def __init__(self, quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self._quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        errors = target - preds
        q = self._quantiles.to(device=preds.device, dtype=preds.dtype).reshape(1, 1, -1)
        loss = torch.maximum(q * errors, (q - 1.0) * errors)
        return loss.mean()
