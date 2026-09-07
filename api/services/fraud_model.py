from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


class FraudModelService:
    """
    Production inference service for the fraud detection model.

    The model is managed and registered in MLflow. In the current
    WSL development environment, the registered model resolves to
    a Windows-backed local artifact, so the service loads that
    verified MLflow model artifact directly.
    """

    MODEL_NAME = "fraud_detection_xgboost"
    MODEL_VERSION = "2"
    THRESHOLD = 0.60

    MODEL_ID = "m-536b08e152104b0dadcdb5f0eb653abd"

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        mlflow_db = project_root / "mlflow.db"

        tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

        mlflow.set_tracking_uri(tracking_uri)

        self.model_uri = (
            f"file://{project_root.as_posix()}/mlruns/1/models/"
            f"{self.MODEL_ID}/artifacts"
        )

        print("Loading MLflow model...")
        print("Tracking URI:", tracking_uri)
        print("Model URI:", self.model_uri)

        self.model = mlflow.pyfunc.load_model(self.model_uri)

        print("MLflow model loaded successfully.")

        self._load_feature_schema()

    def _load_feature_schema(self):
        """
        Extract the feature structure expected by the MLflow model.

        Training used:
            432 original features
            + 432 missingness indicators
            = 864 preprocessing features
        """

        python_model = self.model._model_impl.python_model

        self.preprocessor = python_model.preprocessor

        self.expected_features = list(
            self.preprocessor.feature_names_in_
        )

        self.original_features = [
            column
            for column in self.expected_features
            if not column.endswith("_missing")
        ]

        self.missing_indicator_features = [
            column
            for column in self.expected_features
            if column.endswith("_missing")
        ]

        print(
            "Expected original features:",
            len(self.original_features),
        )

        print(
            "Expected missing indicators:",
            len(self.missing_indicator_features),
        )

        print(
            "Expected total preprocessing features:",
            len(self.expected_features),
        )

    def _prepare_transaction(
        self,
        transaction: dict,
    ) -> pd.DataFrame:
        """
        Convert an incoming transaction dictionary into the exact
        feature structure expected by the trained model.
        """

        df = pd.DataFrame([transaction])

        if "isFraud" in df.columns:
            df = df.drop(columns=["isFraud"])

        if "TransactionID" in df.columns:
            df = df.drop(columns=["TransactionID"])

        for column in self.original_features:
            if column not in df.columns:
                df[column] = np.nan

        df = df[self.original_features].copy()

        for column in self.original_features:
            df[f"{column}_missing"] = df[column].isna().astype(np.int8)

        df = df[self.expected_features].copy()

        return df

    def predict(
        self,
        transaction: dict,
    ) -> dict:
        """
        Run fraud inference for a single transaction.
        """

        features = self._prepare_transaction(transaction)

        prediction_output = self.model.predict(features)

        if isinstance(prediction_output, pd.DataFrame):
            output = prediction_output.iloc[0].to_dict()

            probability = float(
                output.get("fraud_probability", 0.0)
            )

            prediction = bool(
                output.get(
                    "fraud_prediction",
                    probability >= self.THRESHOLD,
                )
            )

            decision = output.get(
                "decision",
                "FRAUD REVIEW"
                if prediction
                else "LEGITIMATE",
            )

        else:
            probability = float(np.asarray(prediction_output).reshape(-1)[0])

            prediction = probability >= self.THRESHOLD

            decision = (
                "FRAUD REVIEW"
                if prediction
                else "LEGITIMATE"
            )

        return {
            "fraud_probability": probability,
            "fraud_prediction": prediction,
            "decision": decision,
            "threshold": self.THRESHOLD,
        }


if __name__ == "__main__":
    service = FraudModelService()

    print()
    print("=" * 60)
    print("FRAUD MODEL SERVICE TEST")
    print("=" * 60)
    print()

    sample_transaction = {
        "TransactionID": 3400379,
        "TransactionDT": 86400,
        "TransactionAmt": 1265.50,
        "ProductCD": "W",
        "card1": 9500,
        "card2": 111.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "credit",
    }

    result = service.predict(sample_transaction)

    print("Prediction result:")
    print(result)

    print()
    print("MODEL SERVICE TEST PASSED.")
