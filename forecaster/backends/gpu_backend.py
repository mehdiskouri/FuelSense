"""CUDA backend for demand forecaster."""

# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportDeprecated=false

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from forecaster.model import DemandTCN, QuantileLoss
from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend


def _cuda_autocast() -> Any:
    torch_amp: Any = getattr(torch, "amp", None)
    if torch_amp is not None:
        amp_autocast: Any = getattr(torch_amp, "autocast", None)
        if amp_autocast is not None:
            return amp_autocast(device_type="cuda", enabled=True)
    torch_autocast: Any = getattr(torch, "autocast", None)
    if torch_autocast is not None:
        return torch_autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _cuda_grad_scaler() -> Any:
    torch_amp: Any = getattr(torch, "amp", None)
    if torch_amp is not None:
        amp_grad_scaler_ctor: Any = getattr(torch_amp, "GradScaler", None)
        if amp_grad_scaler_ctor is not None:
            return amp_grad_scaler_ctor("cuda", enabled=True)
    grad_scaler_ctor: Any = getattr(torch, "GradScaler", None)
    if grad_scaler_ctor is not None:
        return grad_scaler_ctor(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


@register_backend("demand_forecaster", DeviceType.CUDA)
class CUDAForecaster(ComputeBackend):
    device = DeviceType.CUDA

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but CUDA is not available")

        self.cuda_device = torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.cuda_device)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        mem_fraction = float(os.environ.get("FUELSENSE_CUDA_MEMORY_FRACTION", "0.8"))
        try:
            cuda_set_mem_fraction: Any = torch.cuda.set_per_process_memory_fraction
            cuda_set_mem_fraction(mem_fraction, device=self.cuda_device)
        except RuntimeError:
            pass

        self.model: nn.Module | None = DemandTCN().to(self.cuda_device)
        self.model.eval()
        self.loss_fn = QuantileLoss().to(self.cuda_device)

    def warmup(self) -> None:
        if self.model is None:
            return
        with torch.no_grad(), _cuda_autocast():
            dummy = torch.zeros(
                (1, DemandTCN.LOOKBACK, DemandTCN.N_FEATURES), dtype=torch.float32, device=self.cuda_device
            )
            _ = self.model(dummy)
        torch.cuda.synchronize(self.cuda_device)

    def health_check(self) -> dict[str, object]:
        mem_free, mem_total = torch.cuda.mem_get_info(device=self.cuda_device)
        details: dict[str, object] = {
            "device": self.device.value,
            "gpu_name": torch.cuda.get_device_name(self.cuda_device),
            "gpu_memory_free_gb": float(mem_free / (1024**3)),
            "gpu_memory_total_gb": float(mem_total / (1024**3)),
            "model_loaded": bool(self.model is not None),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        }

        try:
            import pynvml

            nvml: Any = pynvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            details["gpu_utilization"] = int(util.gpu)
            nvml.nvmlShutdown()
        except Exception:
            details["gpu_utilization"] = None

        return details

    def load_model(self, state_dict_path: str | Path) -> None:
        if self.model is None:
            self.model = DemandTCN().to(self.cuda_device)

        state_dict = torch.load(Path(state_dict_path), map_location=self.cuda_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        try:
            torch_compile: Any = torch.compile
            self.model = torch_compile(self.model, mode="reduce-overhead")
        except Exception:
            pass

        self.warmup()

    def predict(self, lookback: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        np_in = np.asarray(lookback, dtype=np.float32)
        cpu_tensor = torch.from_numpy(np_in).pin_memory()

        with torch.cuda.stream(self.stream), torch.no_grad(), _cuda_autocast():
            gpu_tensor = cpu_tensor.to(self.cuda_device, non_blocking=True)
            preds = self.model(gpu_tensor)
            preds_cpu = preds.float().detach().cpu()

        self.stream.synchronize()
        return preds_cpu.numpy()

    def train(
        self,
        train_data: np.ndarray,
        train_targets: np.ndarray,
        val_data: np.ndarray,
        val_targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 512,
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

        model = DemandTCN().to(self.cuda_device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_any: Any = optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = _cuda_grad_scaler()

        train_x = torch.as_tensor(train_data, dtype=torch.float32)
        train_y = torch.as_tensor(train_targets, dtype=torch.float32)
        val_x = torch.as_tensor(val_data, dtype=torch.float32, device=self.cuda_device)
        val_y = torch.as_tensor(val_targets, dtype=torch.float32, device=self.cuda_device)

        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        best_val = float("inf")
        best_state_cpu: dict[str, Tensor] | None = None
        stale_epochs = 0

        for _epoch in range(epochs):
            model.train()
            train_losses: list[float] = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.cuda_device, non_blocking=True)
                batch_y = batch_y.to(self.cuda_device, non_blocking=True)

                optimizer_any.zero_grad(set_to_none=True)
                with _cuda_autocast():
                    preds = model(batch_x)
                    target = _expand_targets(batch_y)
                    loss = self.loss_fn(preds, target)

                scaled_loss: Any = scaler.scale(loss)
                scaled_loss.backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            with torch.no_grad(), _cuda_autocast():
                val_preds = model(val_x)
                val_target = _expand_targets(val_y)
                val_loss = float(self.loss_fn(val_preds, val_target).detach().cpu().item())

            epoch_train = float(np.mean(train_losses)) if train_losses else val_loss
            history["train_loss"].append(epoch_train)
            history["val_loss"].append(val_loss)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            if val_loss < best_val:
                best_val = val_loss
                best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            scheduler.step()
            if stale_epochs >= patience:
                break

        if best_state_cpu is not None:
            model.load_state_dict(best_state_cpu)

        self.model = model
        self.model.eval()

        with torch.no_grad(), _cuda_autocast():
            start = perf_counter()
            pred = self.model(val_x)
            torch.cuda.synchronize(self.cuda_device)
            inference_ms = (perf_counter() - start) * 1000
            pred_p50 = pred[:, :, 1]
            val_target = _expand_targets(val_y)
            rmse = float(torch.sqrt(torch.mean((pred_p50 - val_target) ** 2)).detach().cpu().item())

        train_rmse = 0.0
        if train_x.shape[0] > 0:
            with torch.no_grad(), _cuda_autocast():
                full_train_x = train_x.to(self.cuda_device, non_blocking=True)
                full_train_y = train_y.to(self.cuda_device, non_blocking=True)
                train_pred = self.model(full_train_x)
                train_p50 = train_pred[:, :, 1]
                train_target = _expand_targets(full_train_y)
                train_rmse = float(torch.sqrt(torch.mean((train_p50 - train_target) ** 2)).detach().cpu().item())

        return {
            "history": history,
            "best_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "best_val_loss": best_val,
            "epochs_trained": len(history["val_loss"]),
            "training_rmse": train_rmse,
            "validation_rmse": rmse,
            "validation_inference_time_ms": inference_ms,
        }
