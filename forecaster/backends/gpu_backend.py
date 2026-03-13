"""CUDA backend for demand forecaster."""

# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportDeprecated=false

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, suppress
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset

from forecaster.model import DemandTCN, QuantileLoss
from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend

logger = logging.getLogger(__name__)

TARGET_MATRIX_NDIMS = 2

if TYPE_CHECKING:
    from types import ModuleType


def _get_pynvml() -> ModuleType | None:
    try:
        module = import_module("pynvml")
    except ImportError:  # pragma: no cover - optional GPU telemetry dependency
        return None
    return module


class _GradScalerLike(Protocol):
    def scale(self, loss: Tensor) -> _ScaledLossLike: ...

    def unscale_(self, optimizer: AdamW) -> None: ...

    def step(self, optimizer: AdamW) -> None: ...

    def update(self) -> None: ...


class _ScaledLossLike(Protocol):
    def backward(self) -> None: ...


def _cuda_autocast() -> AbstractContextManager[object]:
    torch_amp = getattr(torch, "amp", None)
    if torch_amp is not None:
        amp_autocast = getattr(torch_amp, "autocast", None)
        if callable(amp_autocast):
            return cast("AbstractContextManager[object]", amp_autocast(device_type="cuda", enabled=True))
    torch_autocast = getattr(torch, "autocast", None)
    if callable(torch_autocast):
        return cast("AbstractContextManager[object]", torch_autocast(device_type="cuda", enabled=True))
    return cast("AbstractContextManager[object]", torch.cuda.amp.autocast(enabled=True))


def _cuda_grad_scaler() -> _GradScalerLike:
    torch_amp = getattr(torch, "amp", None)
    if torch_amp is not None:
        amp_grad_scaler_ctor = getattr(torch_amp, "GradScaler", None)
        if callable(amp_grad_scaler_ctor):
            return cast("_GradScalerLike", amp_grad_scaler_ctor("cuda", enabled=True))
    grad_scaler_ctor = getattr(torch, "GradScaler", None)
    if callable(grad_scaler_ctor):
        return cast("_GradScalerLike", grad_scaler_ctor(enabled=True))
    return cast("_GradScalerLike", torch.cuda.amp.GradScaler(enabled=True))


@register_backend("demand_forecaster", DeviceType.CUDA)
class CUDAForecaster(ComputeBackend):
    """GPU-accelerated demand forecaster based on the DemandTCN architecture."""

    device = DeviceType.CUDA

    def __init__(self) -> None:
        """Initialize CUDA resources, model instance, and training configuration."""
        if not torch.cuda.is_available():
            msg = "CUDA backend requested but CUDA is not available"
            raise RuntimeError(msg)

        self.cuda_device = torch.device("cuda:0")
        stream_ctor = cast("Any", torch.cuda.Stream)
        self.stream = cast("torch.cuda.Stream", stream_ctor(device=self.cuda_device))
        self.train_num_workers = int(
            os.environ.get("FUELSENSE_FORECAST_GPU_WORKERS", str(min(4, max((os.cpu_count() or 1) // 2, 0)))),
        )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        mem_fraction = float(os.environ.get("FUELSENSE_CUDA_MEMORY_FRACTION", "0.8"))
        with suppress(RuntimeError):
            torch.cuda.set_per_process_memory_fraction(mem_fraction, device=self.cuda_device)

        self.model: nn.Module | None = DemandTCN().to(self.cuda_device)
        self.model.eval()
        self.loss_fn = QuantileLoss().to(self.cuda_device)

    def warmup(self) -> None:
        """Run a one-shot forward pass to initialize CUDA kernels and memory paths."""
        if self.model is None:
            return
        with torch.no_grad(), _cuda_autocast():
            dummy = torch.zeros(
                (1, DemandTCN.LOOKBACK, DemandTCN.N_FEATURES),
                dtype=torch.float32,
                device=self.cuda_device,
            )
            _ = self.model(dummy)
        torch.cuda.synchronize(self.cuda_device)

    def health_check(self) -> dict[str, object]:
        """Return CUDA device health and optional runtime utilization details."""
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
            pynvml = _get_pynvml()
            if pynvml is None:
                details["gpu_utilization"] = None
            else:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                details["gpu_utilization"] = int(util.gpu)
                pynvml.nvmlShutdown()
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError):
            details["gpu_utilization"] = None

        return details

    def load_model(self, state_dict_path: str | Path) -> None:
        """Load model weights, apply optional compilation, and warm up the backend."""
        if self.model is None:
            self.model = DemandTCN().to(self.cuda_device)

        state_dict = torch.load(Path(state_dict_path), map_location=self.cuda_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        try:
            self.model = cast("nn.Module", torch.compile(self.model, mode="reduce-overhead"))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
            logger.debug("torch.compile unavailable for CUDA forecaster model: %s", exc)

        self.warmup()

    def predict(self, lookback: np.ndarray) -> np.ndarray:
        """Run a CUDA forward pass for a lookback batch and return CPU numpy predictions."""
        if self.model is None:
            msg = "Model is not initialized"
            raise RuntimeError(msg)

        np_in = np.asarray(lookback, dtype=np.float32)
        cpu_tensor = torch.from_numpy(np_in)
        pin_threshold_bytes = int(os.environ.get("FUELSENSE_FORECAST_PIN_MEMORY_MIN_BYTES", "4096"))
        if np_in.nbytes >= max(pin_threshold_bytes, 0):
            cpu_tensor = cpu_tensor.pin_memory()

        with torch.cuda.stream(self.stream), torch.no_grad(), _cuda_autocast():
            gpu_tensor = cpu_tensor.to(self.cuda_device, non_blocking=True)
            preds = self.model(gpu_tensor)
            preds_cpu = preds.detach().cpu()
        return cast("np.ndarray", preds_cpu.numpy())

    @staticmethod
    def _expand_targets(raw: Tensor) -> Tensor:
        if raw.ndim == 1:
            return raw.unsqueeze(1).repeat(1, DemandTCN.HORIZON)
        if raw.ndim == TARGET_MATRIX_NDIMS and raw.shape[1] == DemandTCN.HORIZON:
            return raw
        msg = "targets must have shape [N] or [N, HORIZON]"
        raise ValueError(msg)

    def _build_train_loader(
        self,
        train_x: Tensor,
        train_y: Tensor,
        batch_size: int,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        train_ds = cast("Dataset[tuple[Tensor, Tensor]]", TensorDataset(train_x, train_y))
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=max(self.train_num_workers, 0),
            persistent_workers=self.train_num_workers > 0,
        )

    def _run_train_epoch(
        self,
        model: DemandTCN,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        optimizer: AdamW,
        scaler: _GradScalerLike,
    ) -> list[float]:
        model.train()
        train_losses: list[float] = []
        for batch_x_cpu, batch_y_cpu in train_loader:
            batch_x = batch_x_cpu.to(self.cuda_device, non_blocking=True)
            batch_y = batch_y_cpu.to(self.cuda_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast():
                preds = model(batch_x)
                target = self._expand_targets(batch_y)
                loss = self.loss_fn(preds, target)

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu().item()))
        return train_losses

    def _compute_validation_loss(self, model: DemandTCN, val_x: Tensor, val_y: Tensor) -> float:
        model.eval()
        with torch.no_grad(), _cuda_autocast():
            val_preds = model(val_x)
            val_target = self._expand_targets(val_y)
            return float(self.loss_fn(val_preds, val_target).detach().cpu().item())

    def _compute_rmse_with_timing(self, model: nn.Module, val_x: Tensor, val_y: Tensor) -> tuple[float, float]:
        with torch.no_grad(), _cuda_autocast():
            start = perf_counter()
            pred = model(val_x)
            torch.cuda.synchronize(self.cuda_device)
            inference_ms = (perf_counter() - start) * 1000
            pred_p50 = pred[:, :, 1]
            val_target = self._expand_targets(val_y)
            rmse = float(torch.sqrt(torch.mean((pred_p50 - val_target) ** 2)).detach().cpu().item())
        return rmse, inference_ms

    def _compute_train_rmse(self, model: nn.Module, train_x: Tensor, train_y: Tensor) -> float:
        if train_x.shape[0] == 0:
            return 0.0
        with torch.no_grad(), _cuda_autocast():
            full_train_x = train_x.to(self.cuda_device, non_blocking=True)
            full_train_y = train_y.to(self.cuda_device, non_blocking=True)
            train_pred = model(full_train_x)
            train_p50 = train_pred[:, :, 1]
            train_target = self._expand_targets(full_train_y)
            return float(torch.sqrt(torch.mean((train_p50 - train_target) ** 2)).detach().cpu().item())

    def train(  # noqa: PLR0913
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
        """Train a DemandTCN model on CUDA and return metrics with model artifacts."""
        model = DemandTCN().to(self.cuda_device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = _cuda_grad_scaler()

        train_x = torch.as_tensor(train_data, dtype=torch.float32)
        train_y = torch.as_tensor(train_targets, dtype=torch.float32)
        val_x = torch.as_tensor(val_data, dtype=torch.float32, device=self.cuda_device)
        val_y = torch.as_tensor(val_targets, dtype=torch.float32, device=self.cuda_device)
        train_loader = self._build_train_loader(train_x, train_y, batch_size)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        best_val = float("inf")
        best_state_device: dict[str, Tensor] | None = None
        stale_epochs = 0

        for _epoch in range(epochs):
            train_losses = self._run_train_epoch(model, train_loader, optimizer, scaler)
            val_loss = self._compute_validation_loss(model, val_x, val_y)

            epoch_train = float(np.mean(train_losses)) if train_losses else val_loss
            history["train_loss"].append(epoch_train)
            history["val_loss"].append(val_loss)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            if val_loss < best_val:
                best_val = val_loss
                best_state_device = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            scheduler.step()
            if stale_epochs >= patience:
                break

        if best_state_device is not None:
            model.load_state_dict(best_state_device)

        self.model = model
        self.model.eval()

        rmse, inference_ms = self._compute_rmse_with_timing(self.model, val_x, val_y)
        train_rmse = self._compute_train_rmse(self.model, train_x, train_y)

        return {
            "history": history,
            "best_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "best_val_loss": best_val,
            "epochs_trained": len(history["val_loss"]),
            "training_rmse": train_rmse,
            "validation_rmse": rmse,
            "validation_inference_time_ms": inference_ms,
        }
