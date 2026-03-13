"""Tests for forecast trainer logging and return contract behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from fuelsense_common.compute import DeviceType
from ml_pipeline.training import ForecastTrainer, TrainingDataset

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


FACILITY_ID = 42
EXPECTED_TRAINING_RMSE = 0.75
EXPECTED_VALIDATION_RMSE = 0.82


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _MockBackend:
    def train(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        return {
            "history": {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "lr": [1e-3, 9e-4]},
            "best_state_dict": __import__("forecaster.model", fromlist=["DemandTCN"]).DemandTCN().state_dict(),
            "best_val_loss": 0.9,
            "epochs_trained": 2,
            "training_rmse": 0.75,
            "validation_rmse": 0.82,
        }


def _dataset() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(20, 90, 6)).astype(np.float32)
    y = rng.normal(size=(20, 14)).astype(np.float32)
    return {
        "train_data": x[:10],
        "train_targets": y[:10],
        "val_data": x[10:15],
        "val_targets": y[10:15],
        "test_data": x[15:],
        "test_targets": y[15:],
    }


def test_train_and_register_logs_and_returns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Train-and-register should return backend metrics and emit MLflow logs."""
    mlflow_db = tmp_path / "mlflow.db"
    trainer = ForecastTrainer(tracking_uri=f"sqlite:///{mlflow_db}")
    data = _dataset()

    def _resolve_device() -> DeviceType:
        return DeviceType.CPU

    def _get_backend(_name: str, _device: DeviceType) -> _MockBackend:
        return _MockBackend()

    monkeypatch.setattr("ml_pipeline.training.resolve_device", _resolve_device)
    monkeypatch.setattr("ml_pipeline.training.get_backend", _get_backend)

    logged_params: dict[str, object] = {}
    logged_metrics: list[dict[str, float]] = []
    logged_tags: dict[str, str] = {}

    class _RunInfo:
        run_id = "run-123"

    class _Run:
        info = _RunInfo()

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            _ = exc_type, exc, tb

    def _start_run(run_name: str) -> _Run:
        _ = run_name
        return _Run()

    def _log_params(params: dict[str, object]) -> None:
        logged_params.update(params)

    def _log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        _ = step
        logged_metrics.append(metrics)

    def _log_model(*args: object, **kwargs: object) -> None:
        _ = args, kwargs

    def _set_tags(tags: dict[str, str]) -> None:
        logged_tags.update(tags)

    monkeypatch.setattr("ml_pipeline.training.mlflow.start_run", _start_run)
    monkeypatch.setattr("ml_pipeline.training.mlflow.log_params", _log_params)
    monkeypatch.setattr("ml_pipeline.training.mlflow.log_metrics", _log_metrics)
    monkeypatch.setattr("ml_pipeline.training.mlflow.pytorch.log_model", _log_model)
    monkeypatch.setattr("ml_pipeline.training.mlflow.set_tags", _set_tags)

    result = trainer.train_and_register(
        facility_id=FACILITY_ID,
        dataset=TrainingDataset(
            train_data=data["train_data"],
            train_targets=data["train_targets"],
            val_data=data["val_data"],
            val_targets=data["val_targets"],
            test_data=data["test_data"],
            test_targets=data["test_targets"],
        ),
    )

    _check(result["run_id"] == "run-123", "MLflow run id should be returned")
    _check(result["training_rmse"] == EXPECTED_TRAINING_RMSE, "Training RMSE should match backend output")
    _check(result["validation_rmse"] == EXPECTED_VALIDATION_RMSE, "Validation RMSE should match backend output")
    _check(logged_params["facility_id"] == FACILITY_ID, "Facility id should be logged as a parameter")
    _check(any("train_loss" in item for item in logged_metrics), "Training loss metrics should be logged")
    _check("data_hash" in logged_tags, "Data hash tag should be logged")
