from __future__ import annotations

import torch

from forecaster.model import CausalConv1d, DemandTCN, QuantileLoss


def test_demand_tcn_forward_shapes() -> None:
    model = DemandTCN()
    for batch in (1, 16, 50, 256):
        x = torch.randn(batch, 90, 6)
        y = model(x)
        assert y.shape == (batch, 14, 3)


def test_causal_conv_no_future_leakage() -> None:
    conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)
    conv.conv.weight.data.fill_(1.0)
    assert conv.conv.bias is not None
    conv.conv.bias.data.zero_()

    x1 = torch.zeros(1, 1, 10)
    x2 = x1.clone()
    x2[:, :, -1] = 1000.0

    y1 = conv(x1)
    y2 = conv(x2)

    assert torch.allclose(y1[:, :, :-1], y2[:, :, :-1])


def test_quantile_loss_matches_pinball_formula() -> None:
    loss_fn = QuantileLoss()
    preds = torch.tensor([[[9.0, 10.0, 11.0]]], dtype=torch.float32)
    target = torch.tensor([[10.0]], dtype=torch.float32)
    loss = loss_fn(preds, target)
    assert float(loss.item()) >= 0.0


def test_model_output_quantile_axis_size() -> None:
    model = DemandTCN()
    x = torch.randn(8, 90, 6)
    y = model(x)
    assert y.shape[2] == 3
