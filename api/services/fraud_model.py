import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


class FraudModelService:
    MODEL_NAME = "fraud_detection_xgboost"
    MODEL_VERSION = "6"
    THRESHOLD = 0.60
    MODEL_ID = "m-967f3bf36a4e41de8a7e62e51ba75b3d"

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        mlflow_db = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

        mlflow.set_tracking_uri(tracking_uri)

        self.model_uri = (
            project_root
            / "mlruns"
            / "1"
            / "models"
            / self.MODEL_ID
            / "artifacts"
        ).as_posix()

        print("Loading MLflow model...")
        print("Tracking URI:", tracking_uri)
        print("Model URI:", self.model_uri)

        self.model = mlflow.pyfunc.load_model(self.model_uri)

        print("MLflow model loaded successfully.")

        self._load_feature_schema()

    def _load_feature_schema(self):
        python_model = self.model.unwrap_python_model()

        self.python_model = python_model

        if not hasattr(python_model, "original_features"):
            raise RuntimeError(
                "MLflow Python model does not contain original_features."
            )

        if not hasattr(python_model, "missing_features"):
            raise RuntimeError(
                "MLflow Python model does not contain missing_features."
            )

        if not hasattr(python_model, "expected_features"):
            raise RuntimeError(
                "MLflow Python model does not contain expected_features."
            )

        self.original_features = list(python_model.original_features)
        self.missing_features = list(python_model.missing_features)
        self.expected_features = list(python_model.expected_features)

        if len(self.original_features) != 432:
            raise RuntimeError(
                f"Expected 432 original features, "
                f"got {len(self.original_features)}."
            )

        if len(self.missing_features) != 432:
            raise RuntimeError(
                f"Expected 432 missing indicators, "
                f"got {len(self.missing_features)}."
            )

        if len(self.expected_features) != 864:
            raise RuntimeError(
                f"Expected 864 total model features, "
                f"got {len(self.expected_features)}."
            )

        print("Original features:", len(self.original_features))
        print("Missing indicators:", len(self.missing_features))
        print("Total model features:", len(self.expected_features))

    def _prepare_input(self, transaction: dict) -> pd.DataFrame:
        original = pd.DataFrame(
            [transaction],
            columns=self.original_features,
        )

        for column in self.original_features:
            if column not in transaction:
                original[column] = np.nan

        original = original[self.original_features]

        missing = original.isna().astype(np.int8)
        missing.columns = [
            f"{column}_missing"
            for column in self.original_features
        ]

        features = pd.concat(
            [original, missing],
            axis=1,
        )

        features = features[self.expected_features]

        return features

    def predict(self, transaction: dict) -> dict:
        start_time = time.perf_counter()

        features = self._prepare_input(transaction)

        prediction = self.model.predict(features)

        if isinstance(prediction, pd.DataFrame):
            probability = float(prediction.iloc[0, 0])
        elif isinstance(prediction, pd.Series):
            probability = float(prediction.iloc[0])
        else:
            probability = float(
                np.asarray(prediction).reshape(-1)[0]
            )

        fraud_prediction = probability >= self.THRESHOLD

        if fraud_prediction:
            decision = "FRAUD REVIEW"
        else:
            decision = "LEGITIMATE"

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "fraud_probability": probability,
            "fraud_prediction": fraud_prediction,
            "decision": decision,
            "threshold": self.THRESHOLD,
            "prediction_latency_ms": latency_ms,
        }
