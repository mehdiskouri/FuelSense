"""MLflow-driven training pipeline for demand forecaster models."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pytorch as mlflow_pytorch
import numpy as np
import torch

from forecaster.model import DemandTCN
from fuelsense_common.compute import DeviceType, resolve_device
from fuelsense_common.registry import get_backend


@dataclass(frozen=True)
class TrainerConfig:
    experiment_name: str = "demand-forecaster"
    lookback_days: int = 90
    horizon_days: int = 14
    n_features: int = 6
    hidden_channels: int = 32
    kernel_size: int = 3
    dilations: tuple[int, int, int] = (1, 2, 4)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4


class ForecastTrainer:
    EXPERIMENT_NAME = "demand-forecaster"

    def __init__(self, tracking_uri: str | None = None, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig(experiment_name=self.EXPERIMENT_NAME)
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def _hash_data(self, train_data: np.ndarray) -> str:
        blob = np.asarray(train_data, dtype=np.float32).tobytes()
        return hashlib.sha256(blob).hexdigest()[:12]

    def train_and_register(
        self,
        facility_id: int | None,
        train_data: np.ndarray,
        train_targets: np.ndarray,
        val_data: np.ndarray,
        val_targets: np.ndarray,
        test_data: np.ndarray,
        test_targets: np.ndarray,
    ) -> dict[str, Any]:
        def _validate_arrays(x: np.ndarray, y: np.ndarray, name: str) -> None:
            if x.ndim != 3 or x.shape[1:] != (DemandTCN.LOOKBACK, DemandTCN.N_FEATURES):
                raise ValueError(
                    f"{name}_data must have shape [N, {DemandTCN.LOOKBACK}, {DemandTCN.N_FEATURES}], got {x.shape}",
                )
            if y.ndim == 1:
                if y.shape[0] != x.shape[0]:
                    raise ValueError(f"{name}_targets length must match {name}_data batch")
                return
            if y.ndim == 2 and y.shape == (x.shape[0], DemandTCN.HORIZON):
                return
            raise ValueError(f"{name}_targets must have shape [N] or [N, {DemandTCN.HORIZON}], got {y.shape}")

        def _expand_targets(y: np.ndarray) -> np.ndarray:
            if y.ndim == 1:
                return np.repeat(y[:, None], DemandTCN.HORIZON, axis=1)
            return y

        _validate_arrays(train_data, train_targets, "train")
        _validate_arrays(val_data, val_targets, "val")
        _validate_arrays(test_data, test_targets, "test")
        if train_data.shape[0] == 0:
            raise ValueError("train_data is empty")

        run_name = f"facility_{facility_id}" if facility_id is not None else "global"
        device = resolve_device()
        backend = get_backend("demand_forecaster", device)
        backend_any: Any = backend

        train_result = backend_any.train(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            epochs=100,
            batch_size=512 if device == DeviceType.CUDA else 32,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            patience=10,
        )

        model_for_eval = DemandTCN()
        state_dict = train_result["best_state_dict"]
        model_for_eval.load_state_dict(state_dict)
        model_for_eval.eval()

        with torch.no_grad():
            test_x = torch.as_tensor(test_data, dtype=torch.float32)
            preds = model_for_eval(test_x)
            p50 = preds[:, :, 1].detach().cpu().numpy()
            y_true = _expand_targets(np.asarray(test_targets, dtype=np.float32))
            mse = float(np.asarray(np.mean((p50 - y_true) ** 2) if y_true.size else 0.0, dtype=np.float64).item())
            test_rmse = float(np.sqrt(max(mse, 0.0)))
            denom = np.clip(np.abs(y_true), a_min=1e-6, a_max=None)
            test_mape = float(
                np.asarray(
                    np.mean(np.abs((p50 - y_true) / denom)) * 100 if y_true.size else 0.0, dtype=np.float64,
                ).item(),
            )

        validation_rmse = float(train_result.get("validation_rmse", test_rmse))
        training_rmse = float(train_result.get("training_rmse", validation_rmse))

        with mlflow.start_run(run_name=run_name) as run:
            params: dict[str, str | int | float] = {
                "facility_id": facility_id if facility_id is not None else "global",
                "device": device.value,
                "lookback_days": self.config.lookback_days,
                "horizon_days": self.config.horizon_days,
                "n_features": self.config.n_features,
                "hidden_channels": self.config.hidden_channels,
                "kernel_size": self.config.kernel_size,
                "dilations": json.dumps(self.config.dilations),
                "dropout": self.config.dropout,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "batch_size": 512 if device == DeviceType.CUDA else 32,
                "train_samples": int(train_data.shape[0]),
                "val_samples": int(val_data.shape[0]),
                "test_samples": int(test_data.shape[0]),
            }
            mlflow.log_params(params)

            history = train_result.get("history", {})
            train_losses = list(history.get("train_loss", []))
            val_losses = list(history.get("val_loss", []))
            lrs = list(history.get("lr", []))
            for idx in range(min(len(train_losses), len(val_losses), len(lrs))):
                mlflow.log_metrics(
                    {
                        "train_loss": float(train_losses[idx]),
                        "val_loss": float(val_losses[idx]),
                        "learning_rate": float(lrs[idx]),
                    },
                    step=idx,
                )

            mlflow.log_metrics(
                {
                    "test_rmse": test_rmse,
                    "test_mape": test_mape,
                    "training_rmse": training_rmse,
                    "validation_rmse": validation_rmse,
                    "best_val_loss": float(train_result.get("best_val_loss", 0.0)),
                    "epochs_trained": float(train_result.get("epochs_trained", 0)),
                },
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                torch.save(state_dict, model_path)
                model_artifact = DemandTCN()
                model_artifact.load_state_dict(torch.load(model_path, map_location="cpu"))
                registered_name = (
                    f"demand-forecaster-{facility_id}" if facility_id is not None else "demand-forecaster-global"
                )
                log_model_any: Any = mlflow_pytorch.log_model
                log_model_any(model_artifact, artifact_path="model", registered_model_name=registered_name)

            mlflow.set_tags(
                {
                    "device": device.value,
                    "facility_id": str(facility_id) if facility_id is not None else "global",
                    "data_hash": self._hash_data(train_data),
                },
            )

            run_id = run.info.run_id

        return {
            "facility_id": facility_id,
            "run_id": run_id,
            "registered_model_name": f"demand-forecaster-{facility_id}"
            if facility_id is not None
            else "demand-forecaster-global",
            "device": device.value,
            "training_rmse": training_rmse,
            "validation_rmse": validation_rmse,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "best_val_loss": float(train_result.get("best_val_loss", 0.0)),
            "epochs_trained": int(train_result.get("epochs_trained", 0)),
        }
