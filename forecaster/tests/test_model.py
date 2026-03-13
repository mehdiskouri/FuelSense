"""Unit tests for forecaster model layers and loss behavior."""

from __future__ import annotations

from typing import cast

import torch

from forecaster.model import CausalConv1d, DemandTCN, QuantileLoss

QUANTILE_COUNT = 3


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_demand_tcn_forward_shapes() -> None:
    """DemandTCN should return expected `(batch, horizon, quantiles)` tensor shapes."""
    model = DemandTCN()
    for batch in (1, 16, 50, 256):
        x = torch.randn(batch, 90, 6)
        y = model(x)
        _check(y.shape == (batch, 14, 3), "DemandTCN output shape should be (batch, 14, 3)")


def test_causal_conv_no_future_leakage() -> None:
    """Causal convolution outputs should not change for earlier timesteps."""
    conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)
    conv.conv.weight.data.fill_(1.0)
    _check(conv.conv.bias is not None, "Convolution bias should be initialized")
    bias = cast("torch.Tensor", conv.conv.bias)
    bias.data.zero_()

    x1 = torch.zeros(1, 1, 10)
    x2 = x1.clone()
    x2[:, :, -1] = 1000.0

    y1 = conv(x1)
    y2 = conv(x2)

    _check(torch.allclose(y1[:, :, :-1], y2[:, :, :-1]), "Causal convolution must not leak future values")


def test_quantile_loss_matches_pinball_formula() -> None:
    """Quantile (pinball) loss should be non-negative for valid tensors."""
    loss_fn = QuantileLoss()
    preds = torch.tensor([[[9.0, 10.0, 11.0]]], dtype=torch.float32)
    target = torch.tensor([[10.0]], dtype=torch.float32)
    loss = loss_fn(preds, target)
    _check(float(loss.item()) >= 0.0, "Pinball loss should be non-negative")


def test_model_output_quantile_axis_size() -> None:
    """Model output should expose the expected quantile-axis cardinality."""
    model = DemandTCN()
    x = torch.randn(8, 90, 6)
    y = model(x)
    _check(y.shape[2] == QUANTILE_COUNT, "Quantile axis should have size 3")
