from pathlib import Path
import time

import mlflow
import numpy as np
import pandas as pd


class FraudModelService:
    MODEL_NAME = "fraud_detection_xgboost"
    MODEL_VERSION = "4"
    THRESHOLD = 0.60
    MODEL_ID = "m-84de973f844f4ce5b067e7d54895a1ba"

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        mlflow_db = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

        mlflow.set_tracking_uri(tracking_uri)

        # Use the exact model path that was already confirmed
        # to load successfully under Python 3.13.
        self.model_uri = (
            "C:/Users/mohda/real-time-fraud-detection/"
            "mlruns/1/models/"
            f"{self.MODEL_ID}/artifacts"
        )

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

        self.original_features = list(
            python_model.original_features
        )

        self.missing_features = list(
            python_model.missing_features
        )

        self.expected_features = list(
            python_model.expected_features
        )

        print(
            "Original features:",
            len(self.original_features)
        )

        print(
            "Missing indicators:",
            len(self.missing_features)
        )

        print(
            "Total model features:",
            len(self.expected_features)
        )

        if len(self.original_features) != 432:
            raise RuntimeError(
                f"Expected 432 original features, "
                f"found {len(self.original_features)}."
            )

        if len(self.missing_features) != 432:
            raise RuntimeError(
                f"Expected 432 missing indicators, "
                f"found {len(self.missing_features)}."
            )

        if len(self.expected_features) != 864:
            raise RuntimeError(
                f"Expected 864 total model features, "
                f"found {len(self.expected_features)}."
            )

    def _prepare_input(self, transaction: dict) -> pd.DataFrame:
        input_data = {
            key: value
            for key, value in transaction.items()
            if key not in {
                "isFraud",
                "TransactionID",
            }
        }

        original_data = {
            feature: input_data.get(feature, np.nan)
            for feature in self.original_features
        }

        original_df = pd.DataFrame(
            [original_data],
            columns=self.original_features,
        )

        missing_data = {}

        for feature in self.original_features:
            indicator_name = f"{feature}_missing"

            missing_data[indicator_name] = int(
                original_df[feature]
                .isna()
                .iloc[0]
            )

        missing_df = pd.DataFrame(
            [missing_data],
            columns=self.missing_features,
        )

        model_input = pd.concat(
            [
                original_df,
                missing_df,
            ],
            axis=1,
        )

        model_input = model_input[
            self.expected_features
        ]

        return model_input

    def predict(self, transaction: dict) -> dict:
        start_time = time.perf_counter()

        model_input = self._prepare_input(
            transaction
        )

        probabilities = self.model.predict(
            model_input
        )

        fraud_probability = float(
            np.asarray(probabilities)
            .reshape(-1)[0]
        )

        fraud_prediction = (
            fraud_probability >= self.THRESHOLD
        )

        decision = (
            "FRAUD REVIEW"
            if fraud_prediction
            else "LEGITIMATE"
        )

        prediction_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "fraud_probability": fraud_probability,
            "fraud_prediction": bool(
                fraud_prediction
            ),
            "decision": decision,
            "threshold": self.THRESHOLD,
            "prediction_latency_ms": (
                prediction_latency_ms
            ),
        }


def main():
    print("=" * 60)
    print("FRAUD MODEL SERVICE TEST")
    print("=" * 60)

    service = FraudModelService()

    sample_transaction = {
        "TransactionID": 3400379,
        "TransactionDT": 86400,
        "TransactionAmt": 1265.50,
        "ProductCD": "W",
        "card1": 9500,
        "card2": 321,
        "card3": 150,
        "card4": "visa",
        "card5": 226,
        "card6": "credit",
        "addr1": 123,
        "addr2": 87,
        "dist1": 10.0,
        "dist2": 20.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
    }

    print()
    print("Running prediction...")

    try:
        result = service.predict(
            sample_transaction
        )

        print()
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)

        print(
            f"Fraud probability: "
            f"{result['fraud_probability']:.6f}"
        )

        print(
            f"Fraud prediction: "
            f"{result['fraud_prediction']}"
        )

        print(
            f"Decision: "
            f"{result['decision']}"
        )

        print(
            f"Threshold: "
            f"{result['threshold']:.2f}"
        )

        print(
            f"Prediction latency: "
            f"{result['prediction_latency_ms']:.2f} ms"
        )

        print()
        print("=" * 60)
        print("MODEL SERVICE TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as exc:
        print()
        print("=" * 60)
        print("MODEL SERVICE TEST FAILED")
        print("=" * 60)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise


if __name__ == "__main__":
    main()
