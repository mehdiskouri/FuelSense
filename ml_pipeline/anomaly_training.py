# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from anomaly.detector import AnomalyDetector


class AnomalyTrainer:
    EXPERIMENT_NAME = "anomaly-detector"

    def __init__(self, tracking_uri: str | None = None, random_state: int = 42) -> None:
        self.random_state = random_state
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.EXPERIMENT_NAME)

    def generate_synthetic_anomalies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        normal_count = 17350

        normal = np.column_stack(
            [
                rng.normal(0.0, 1.0, normal_count),
                rng.normal(1.0, 0.4, normal_count),
                rng.normal(0.0, 0.15, normal_count),
                rng.normal(0.0, 1.5, normal_count),
                rng.integers(0, 7, normal_count),
                rng.uniform(6.0, 84.0, normal_count),
                rng.uniform(0.2, 0.95, normal_count),
            ]
        )

        per_type = 180
        typed_blocks: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        # 0 LEAK: sustained high positive deviation, low inventory.
        leak = np.column_stack(
            [
                rng.normal(4.8, 0.7, per_type),
                rng.normal(4.0, 0.7, per_type),
                rng.normal(0.35, 0.10, per_type),
                rng.normal(1.5, 1.2, per_type),
                rng.integers(0, 7, per_type),
                rng.uniform(24.0, 168.0, per_type),
                rng.uniform(0.02, 0.25, per_type),
            ]
        )
        typed_blocks.append(leak)
        labels.append(np.full(per_type, 0, dtype=np.int32))

        # 1 THEFT: large spikes shortly after delivery events.
        theft = np.column_stack(
            [
                rng.normal(5.1, 0.9, per_type),
                rng.normal(3.5, 0.8, per_type),
                rng.normal(0.55, 0.12, per_type),
                rng.normal(0.0, 1.0, per_type),
                rng.integers(0, 7, per_type),
                rng.uniform(0.2, 14.0, per_type),
                rng.uniform(0.05, 0.4, per_type),
            ]
        )
        typed_blocks.append(theft)
        labels.append(np.full(per_type, 1, dtype=np.int32))

        # 2 EQUIPMENT_DEGRADATION: moderate but persistent drift with heat-related residuals.
        equipment = np.column_stack(
            [
                rng.normal(3.6, 0.5, per_type),
                rng.normal(2.8, 0.6, per_type),
                rng.normal(0.2, 0.08, per_type),
                rng.normal(4.0, 1.1, per_type),
                rng.integers(0, 7, per_type),
                rng.uniform(36.0, 180.0, per_type),
                rng.uniform(0.15, 0.5, per_type),
            ]
        )
        typed_blocks.append(equipment)
        labels.append(np.full(per_type, 2, dtype=np.int32))

        # 3 DEMAND_SHIFT: day-of-week and demand regime changes.
        demand_shift = np.column_stack(
            [
                rng.normal(3.8, 0.8, per_type),
                rng.normal(3.0, 0.9, per_type),
                rng.normal(0.7, 0.2, per_type),
                rng.normal(0.0, 1.0, per_type),
                rng.choice([0.0, 1.0, 5.0, 6.0], per_type),
                rng.uniform(12.0, 96.0, per_type),
                rng.uniform(0.2, 0.8, per_type),
            ]
        )
        typed_blocks.append(demand_shift)
        labels.append(np.full(per_type, 3, dtype=np.int32))

        # 4 SENSOR_FAULT: extreme residuals and implausible inventory percentages.
        sensor_fault = np.column_stack(
            [
                rng.normal(6.2, 0.9, per_type),
                rng.normal(4.6, 1.0, per_type),
                rng.normal(0.05, 0.2, per_type),
                rng.normal(9.0, 1.8, per_type),
                rng.integers(0, 7, per_type),
                rng.uniform(1.0, 200.0, per_type),
                rng.choice([0.0, 1.1, 1.3], per_type),
            ]
        )
        typed_blocks.append(sensor_fault)
        labels.append(np.full(per_type, 4, dtype=np.int32))

        anomaly_typed = np.vstack(typed_blocks).astype(np.float32)
        y_typed = np.concatenate(labels)

        X_all = np.vstack([normal.astype(np.float32), anomaly_typed])
        y_if = np.concatenate([np.zeros(normal_count, dtype=np.int32), np.ones(anomaly_typed.shape[0], dtype=np.int32)])
        return X_all, y_if, y_typed

    @staticmethod
    def train_isolation_forest(X: np.ndarray) -> IsolationForest:
        model = IsolationForest(
            n_estimators=AnomalyDetector.N_ESTIMATORS,
            contamination=AnomalyDetector.CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        return model

    @staticmethod
    def train_type_classifier(X_anomalies: np.ndarray, y_labels: np.ndarray) -> RandomForestClassifier:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_anomalies, y_labels)
        return clf

    def train_and_register(self, artifact_dir: str | Path | None = None) -> dict[str, Any]:
        X_all, y_if, y_typed = self.generate_synthetic_anomalies()
        anomaly_count = int(np.sum(y_if == 1))
        X_anomalies = X_all[-anomaly_count:]

        X_if_train, X_if_test, _y_if_train, y_if_test = train_test_split(
            X_all,
            y_if,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_if,
        )

        X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
            X_anomalies,
            y_typed,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_typed,
        )

        iforest = self.train_isolation_forest(X_if_train)
        classifier = self.train_type_classifier(X_cls_train, y_cls_train)

        y_if_pred = (iforest.predict(X_if_test) == -1).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(y_if_test, y_if_pred, average="binary", zero_division=0)

        cls_accuracy = float(np.mean(classifier.predict(X_cls_test) == y_cls_test))

        artifact_root = Path(artifact_dir or os.environ.get("ANOMALY_ARTIFACT_DIR", "/tmp/fuelsense-anomaly"))
        artifact_root.mkdir(parents=True, exist_ok=True)
        forest_path = artifact_root / "isolation_forest.joblib"
        classifier_path = artifact_root / "type_classifier.joblib"
        joblib.dump(iforest, forest_path)
        joblib.dump(classifier, classifier_path)

        with mlflow.start_run(run_name="anomaly-detector-train") as run:
            mlflow.log_params(
                {
                    "n_features": 7,
                    "n_estimators_if": AnomalyDetector.N_ESTIMATORS,
                    "contamination": AnomalyDetector.CONTAMINATION,
                    "normal_samples": int(np.sum(y_if == 0)),
                    "anomaly_samples": anomaly_count,
                    "type_count": 5,
                    "if_train_size": int(X_if_train.shape[0]),
                    "if_test_size": int(X_if_test.shape[0]),
                    "cls_train_size": int(X_cls_train.shape[0]),
                    "cls_test_size": int(X_cls_test.shape[0]),
                    "rf_estimators": 300,
                    "rf_max_depth": 10,
                    "rf_min_samples_leaf": 2,
                }
            )
            mlflow.log_metrics(
                {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "classifier_accuracy": cls_accuracy,
                }
            )
            mlflow.log_artifact(str(forest_path), artifact_path="models")
            mlflow.log_artifact(str(classifier_path), artifact_path="models")
            run_id = run.info.run_id

        return {
            "run_id": run_id,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "classifier_accuracy": cls_accuracy,
            "forest_path": str(forest_path),
            "classifier_path": str(classifier_path),
        }
