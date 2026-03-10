"""CPU backend for demand forecaster."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from forecaster.model import DemandTCN, QuantileLoss
from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend


@register_backend("demand_forecaster", DeviceType.CPU)
class CPUForecaster(ComputeBackend):
    device = DeviceType.CPU

    def __init__(self) -> None:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("medium")
        self.train_num_workers = int(
            os.environ.get("FUELSENSE_FORECAST_CPU_WORKERS", str(min(4, max((os.cpu_count() or 1) - 1, 0))))
        )
        self.model: DemandTCN | None = DemandTCN()
        self.model.eval()
        self.loss_fn = QuantileLoss()

    def warmup(self) -> None:
        with torch.no_grad():
            dummy = torch.zeros((1, DemandTCN.LOOKBACK, DemandTCN.N_FEATURES), dtype=torch.float32)
            if self.model is not None:
                _ = self.model(dummy)

    def health_check(self) -> dict[str, object]:
        return {
            "device": self.device.value,
            "threads": torch.get_num_threads(),
            "mkl_available": bool(torch.backends.mkl.is_available()),
            "model_loaded": bool(self.model is not None),
        }

    def load_model(self, state_dict_path: str | Path) -> None:
        payload = torch.load(Path(state_dict_path), map_location="cpu")
        if self.model is None:
            self.model = DemandTCN()
        self.model.load_state_dict(payload)
        self.model.eval()
        self.warmup()

    def predict(self, lookback: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(lookback, dtype=torch.float32)
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        with torch.no_grad():
            preds = self.model(x)
        return preds.detach().cpu().numpy()

    def train(
        self,
        train_data: np.ndarray,
        train_targets: np.ndarray,
        val_data: np.ndarray,
        val_targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
    ) -> dict[str, Any]:
        def _expand_targets(raw: Tensor) -> Tensor:
            if raw.ndim == 1:
                return raw.unsqueeze(1).repeat(1, DemandTCN.HORIZON)
            if raw.ndim == 2 and raw.shape[1] == DemandTCN.HORIZON:
                return raw
            raise ValueError("targets must have shape [N] or [N, HORIZON]")

        model = DemandTCN()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_any: Any = optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

        train_x = torch.as_tensor(train_data, dtype=torch.float32)
        train_y = torch.as_tensor(train_targets, dtype=torch.float32)
        val_x = torch.as_tensor(val_data, dtype=torch.float32)
        val_y = torch.as_tensor(val_targets, dtype=torch.float32)

        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max(self.train_num_workers, 0),
            persistent_workers=self.train_num_workers > 0,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        best_val = float("inf")
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0

        for _epoch in range(epochs):
            model.train()
            train_losses: list[float] = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                preds = model(batch_x)
                target = _expand_targets(batch_y)
                loss = self.loss_fn(preds, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_any.step()
                train_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                val_preds = model(val_x)
                val_target = _expand_targets(val_y)
                val_loss = float(self.loss_fn(val_preds, val_target).detach().cpu().item())

            epoch_train = float(np.mean(train_losses)) if train_losses else val_loss
            history["train_loss"].append(epoch_train)
            history["val_loss"].append(val_loss)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            scheduler.step()
            if stale_epochs >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.model.eval()

        with torch.no_grad():
            start = perf_counter()
            pred = self.model(val_x)
            inference_ms = (perf_counter() - start) * 1000
            pred_p50 = pred[:, :, 1]
            val_target = _expand_targets(val_y)
            rmse = float(torch.sqrt(torch.mean((pred_p50 - val_target) ** 2)).item())

        train_rmse = 0.0
        if train_x.shape[0] > 0:
            with torch.no_grad():
                train_pred = self.model(train_x)
                train_p50 = train_pred[:, :, 1]
                train_target = _expand_targets(train_y)
                train_rmse = float(torch.sqrt(torch.mean((train_p50 - train_target) ** 2)).item())

        return {
            "history": history,
            "best_state_dict": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "best_val_loss": best_val,
            "epochs_trained": len(history["val_loss"]),
            "training_rmse": train_rmse,
            "validation_rmse": rmse,
            "validation_inference_time_ms": inference_ms,
        }
