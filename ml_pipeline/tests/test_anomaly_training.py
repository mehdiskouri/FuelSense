"""Tests for anomaly trainer synthetic generation, model fitting, and MLflow logging."""

# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from ml_pipeline.anomaly_training import AnomalyTrainer

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


FEATURE_COUNT = 7
SAMPLE_SIZE = 8
MIN_PRECISION = 0.80
MIN_RECALL = 0.70
EXPECTED_ARTIFACTS = 2


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def test_generate_synthetic_anomalies_shape_and_labels() -> None:
    """Synthetic anomaly generation should return aligned feature and label tensors."""
    trainer = AnomalyTrainer(tracking_uri="sqlite:////tmp/fuelsense-mlflow-anomaly-test.db")
    x, y_if, y_typed = trainer.generate_synthetic_anomalies()

    _check(x.shape[1] == FEATURE_COUNT)
    _check(x.shape[0] == y_if.shape[0])
    _check(int(np.sum(y_if == 1)) == y_typed.shape[0])
    _check(set(np.unique(y_typed).tolist()) == {0, 1, 2, 3, 4})


def test_train_models_expose_required_interfaces() -> None:
    """Trainer models should expose expected predict and score interfaces."""
    trainer = AnomalyTrainer(tracking_uri="sqlite:////tmp/fuelsense-mlflow-anomaly-test.db")
    x, y_if, y_typed = trainer.generate_synthetic_anomalies()
    anomaly_count = int(np.sum(y_if == 1))
    x_anom = x[-anomaly_count:]

    if_model = trainer.train_isolation_forest(x)
    cls = trainer.train_type_classifier(x_anom, y_typed)

    sample = x[:SAMPLE_SIZE]
    _check(if_model.predict(sample).shape == (SAMPLE_SIZE,))
    _check(if_model.decision_function(sample).shape == (SAMPLE_SIZE,))
    _check(cls.predict(sample).shape == (SAMPLE_SIZE,))
    _check(cls.predict_proba(sample).shape[0] == SAMPLE_SIZE)


def test_train_and_register_logs_and_quality(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end train-and-register should log metrics and artifacts to MLflow."""
    trainer = AnomalyTrainer(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    logged_params: dict[str, object] = {}
    logged_metrics: dict[str, float] = {}
    logged_artifacts: list[str] = []

    class _RunInfo:
        run_id = "anomaly-run-1"

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

    def _start_run(*, run_name: str | None = None) -> _Run:
        _ = run_name
        return _Run()

    def _log_artifact(path: str, *, artifact_path: str | None = None) -> None:
        _ = artifact_path
        logged_artifacts.append(str(path))

    monkeypatch.setattr(
        "ml_pipeline.anomaly_training.mlflow.start_run",
        _start_run,
    )
    monkeypatch.setattr("ml_pipeline.anomaly_training.mlflow.log_params", logged_params.update)
    monkeypatch.setattr("ml_pipeline.anomaly_training.mlflow.log_metrics", logged_metrics.update)
    monkeypatch.setattr(
        "ml_pipeline.anomaly_training.mlflow.log_artifact",
        _log_artifact,
    )

    result = trainer.train_and_register(artifact_dir=tmp_path)

    _check(result["run_id"] == "anomaly-run-1")
    _check(logged_params["n_features"] == FEATURE_COUNT)
    _check("precision" in logged_metrics)
    _check("recall" in logged_metrics)
    _check(result["precision"] > MIN_PRECISION)
    _check(result["recall"] > MIN_RECALL)
    _check(len(logged_artifacts) == EXPECTED_ARTIFACTS)
