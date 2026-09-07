import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = (
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

MODEL_URI = (
    "models:/m-84de973f844f4ce5b067e7d54895a1ba"
)

THRESHOLD = 0.60


# ============================================================
# FRAUD MODEL INFERENCE
# ============================================================

class FraudModelInference:
    """
    Fraud detection inference service.

    The MLflow wrapper owns the complete feature engineering
    logic.

    Incoming data:
        Original transaction features

    MLflow wrapper internally creates:
        432 original features
        432 missingness indicators

    Total model features:
        864
    """

    def __init__(self):

        print("Loading fraud detection model...")
        print(f"MLflow URI: {MODEL_URI}")

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        self.model = mlflow.pyfunc.load_model(
            MODEL_URI
        )

        # Access the custom FraudDetectionModel
        # stored inside MLflow.
        self.python_model = (
            self.model.unwrap_python_model()
        )

        # IMPORTANT:
        # Use the feature schema defined by the
        # MLflow wrapper itself.
        self.original_features = list(
            self.python_model.original_features
        )

        self.missing_features = list(
            self.python_model.missing_features
        )

        self.expected_features = list(
            self.python_model.expected_features
        )

        print(
            "Fraud detection model loaded successfully."
        )

        print(
            f"Original features: "
            f"{len(self.original_features)}"
        )

        print(
            f"Missing indicators: "
            f"{len(self.missing_features)}"
        )

        print(
            f"Total model features: "
            f"{len(self.expected_features)}"
        )

    # ========================================================
    # PREPARE TRANSACTION
    # ========================================================

    def prepare_transaction(
        self,
        transaction: dict,
    ) -> pd.DataFrame:
        """
        Prepare an incoming transaction.

        The MLflow model itself creates missingness
        indicators, so this method only prepares the
        original transaction feature columns.
        """

        dataframe = pd.DataFrame(
            [transaction]
        )

        # Remove target and identifier fields.
        dataframe = dataframe.drop(
            columns=[
                "TransactionID",
                "isFraud",
            ],
            errors="ignore",
        )

        # Construct all original features at once.
        prepared_data = {
            feature: (
                dataframe[feature].iloc[0]
                if feature in dataframe.columns
                else np.nan
            )
            for feature in self.original_features
        }

        return pd.DataFrame(
            [prepared_data],
            columns=self.original_features,
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        transaction: dict,
    ) -> dict:

        dataframe = self.prepare_transaction(
            transaction
        )

        result = self.model.predict(
            dataframe
        )

        if isinstance(
            result,
            pd.DataFrame,
        ):

            row = result.iloc[0]

            probability = float(
                row["fraud_probability"]
            )

            prediction = bool(
                row["fraud_prediction"]
            )

            decision = str(
                row["decision"]
            )

        else:

            probability = float(
                result[0]
            )

            prediction = (
                probability >= THRESHOLD
            )

            decision = (
                "FRAUD REVIEW"
                if prediction
                else "LEGITIMATE"
            )

        return {
            "fraud_probability": probability,
            "fraud_prediction": prediction,
            "decision": decision,
            "threshold": THRESHOLD,
        }


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("FRAUD MODEL INFERENCE TEST")
    print("=" * 70)
    print()

    inference = FraudModelInference()

    # --------------------------------------------------------
    # Example transaction.
    # --------------------------------------------------------

    transaction = {
        "TransactionID": 3400379,
        "TransactionDT": 10438003,
        "TransactionAmt": 1265.50,
        "ProductCD": "W",
        "card1": 9500,
        "card2": 321,
        "card3": 150,
        "card4": "visa",
        "card5": 226,
        "card6": "debit",
        "addr1": 100,
        "addr2": 87,
        "dist1": 10.0,
        "dist2": None,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
    }

    print()
    print("Preparing transaction...")
    print()

    result = inference.predict(
        transaction
    )

    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        f"Fraud probability: "
        f"{result['fraud_probability']:.6f}"
    )

    print(
        f"Fraud prediction:  "
        f"{result['fraud_prediction']}"
    )

    print(
        f"Decision:           "
        f"{result['decision']}"
    )

    print(
        f"Threshold:          "
        f"{result['threshold']:.2f}"
    )

    print("=" * 70)
    print()

    print(
        "Model inference test completed successfully."
    )


if __name__ == "__main__":
    main()
