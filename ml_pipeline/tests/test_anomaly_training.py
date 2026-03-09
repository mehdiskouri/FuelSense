# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.anomaly_training import AnomalyTrainer


def test_generate_synthetic_anomalies_shape_and_labels() -> None:
    trainer = AnomalyTrainer(tracking_uri="sqlite:////tmp/fuelsense-mlflow-anomaly-test.db")
    X, y_if, y_typed = trainer.generate_synthetic_anomalies()

    assert X.shape[1] == 7
    assert X.shape[0] == y_if.shape[0]
    assert int(np.sum(y_if == 1)) == y_typed.shape[0]
    assert set(np.unique(y_typed).tolist()) == {0, 1, 2, 3, 4}


def test_train_models_expose_required_interfaces() -> None:
    trainer = AnomalyTrainer(tracking_uri="sqlite:////tmp/fuelsense-mlflow-anomaly-test.db")
    X, y_if, y_typed = trainer.generate_synthetic_anomalies()
    anomaly_count = int(np.sum(y_if == 1))
    X_anom = X[-anomaly_count:]

    if_model = trainer.train_isolation_forest(X)
    cls = trainer.train_type_classifier(X_anom, y_typed)

    sample = X[:8]
    assert if_model.predict(sample).shape == (8,)
    assert if_model.decision_function(sample).shape == (8,)
    assert cls.predict(sample).shape == (8,)
    assert cls.predict_proba(sample).shape[0] == 8


def test_train_and_register_logs_and_quality(monkeypatch: Any, tmp_path: Path) -> None:
    trainer = AnomalyTrainer(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    logged_params: dict[str, Any] = {}
    logged_metrics: dict[str, float] = {}
    logged_artifacts: list[str] = []

    class _RunInfo:
        run_id = "anomaly-run-1"

    class _Run:
        info = _RunInfo()

        def __enter__(self) -> "_Run":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = exc_type, exc, tb

    monkeypatch.setattr("ml_pipeline.anomaly_training.mlflow.start_run", lambda run_name: _Run())
    monkeypatch.setattr("ml_pipeline.anomaly_training.mlflow.log_params", lambda p: logged_params.update(p))
    monkeypatch.setattr("ml_pipeline.anomaly_training.mlflow.log_metrics", lambda m: logged_metrics.update(m))
    monkeypatch.setattr("ml_pipeline.anomaly_training.mlflow.log_artifact", lambda path, artifact_path=None: logged_artifacts.append(str(path)))

    result = trainer.train_and_register(artifact_dir=tmp_path)

    assert result["run_id"] == "anomaly-run-1"
    assert logged_params["n_features"] == 7
    assert "precision" in logged_metrics
    assert "recall" in logged_metrics
    assert result["precision"] > 0.80
    assert result["recall"] > 0.70
    assert len(logged_artifacts) == 2
