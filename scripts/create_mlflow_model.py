from pathlib import Path
import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "xgboost_missing_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "xgboost_missing_preprocessor.joblib"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "fraud-detection-modeling"
REGISTERED_MODEL_NAME = "fraud_detection_xgboost"


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.60


# ============================================================
# CUSTOM MLflow PYFUNC MODEL
# ============================================================

class FraudDetectionModel(mlflow.pyfunc.PythonModel):
    """
    Complete fraud detection inference pipeline.

    Pipeline:

        Raw transaction
              ↓
        Missingness indicators
              ↓
        Saved preprocessor
              ↓
        XGBoost model
              ↓
        Fraud probability
    """

    def load_context(self, context):

        self.model = joblib.load(context.artifacts["model"])

        self.preprocessor = joblib.load(
            context.artifacts["preprocessor"]
        )

        self.threshold = THRESHOLD

    def _add_missing_indicators(self, df):

        df = df.copy()

        original_columns = list(df.columns)

        for column in original_columns:
            indicator_name = f"{column}_missing"

            if indicator_name not in df.columns:
                df[indicator_name] = df[column].isna().astype("int8")

        return df

    def predict(self, context, model_input):

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        df = model_input.copy()

        # ----------------------------------------------------
        # Remove columns that are not model features
        # ----------------------------------------------------

        columns_to_drop = []

        for column in ["isFraud", "TransactionID"]:
            if column in df.columns:
                columns_to_drop.append(column)

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # ----------------------------------------------------
        # Add missingness indicators
        # ----------------------------------------------------

        df = self._add_missing_indicators(df)

        # ----------------------------------------------------
        # Transform features
        # ----------------------------------------------------

        X = self.preprocessor.transform(df)

        # ----------------------------------------------------
        # Generate fraud probability
        # ----------------------------------------------------

        probabilities = self.model.predict_proba(X)[:, 1]

        # ----------------------------------------------------
        # Apply production decision threshold
        # ----------------------------------------------------

        predictions = (probabilities >= self.threshold).astype(int)

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "fraud_prediction": predictions,
                "decision": np.where(
                    predictions == 1,
                    "FRAUD REVIEW",
                    "LEGITIMATE"
                ),
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING MLflow DEPLOYABLE FRAUD MODEL")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nChecking model files...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    print("XGBoost model:", MODEL_PATH)
    print("Preprocessor:", PREPROCESSOR_PATH)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

    print("\nConnecting to MLflow...")
    print("Tracking URI:", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Experiment ID:", experiment_id)

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    print("\nStarting MLflow run...")

    with mlflow.start_run(
        run_name="xgboost-missingness-deployable"
    ) as run:

        print("Run ID:", run.info.run_id)

        # ----------------------------------------------------
        # Log model configuration
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "feature_strategy": (
                    "original_features_plus_missing_indicators"
                ),
                "original_features": 432,
                "missing_indicators": 432,
                "total_features_before_encoding": 864,
                "threshold": THRESHOLD,
                "deployment_model": True,
            }
        )

        # ----------------------------------------------------
        # Create sample input signature
        # ----------------------------------------------------

        sample_data = pd.DataFrame(
            {
                "TransactionDT": [100000],
                "TransactionAmt": [100.0],
                "ProductCD": ["W"],
                "card1": [1000],
                "card2": [100.0],
                "card3": [150.0],
                "card4": ["visa"],
                "card5": [226.0],
                "card6": ["debit"],
                "addr1": [100.0],
                "addr2": [87.0],
                "dist1": [1.0],
                "dist2": [np.nan],
                "P_emaildomain": ["gmail.com"],
                "R_emaildomain": [np.nan],
            }
        )

        # ----------------------------------------------------
        # Log deployable MLflow model
        # ----------------------------------------------------

        print("\nLogging deployable MLflow model...")

        model_info = mlflow.pyfunc.log_model(
            artifact_path="fraud_detection_model",
            python_model=FraudDetectionModel(),
            artifacts={
                "model": str(MODEL_PATH),
                "preprocessor": str(PREPROCESSOR_PATH),
            },
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("\nModel logged successfully.")

        print("Model URI:")
        print(model_info.model_uri)

        print("\nModel artifact URI:")
        print(model_info.artifact_path)

        print("\nRegistered model:")
        print(REGISTERED_MODEL_NAME)

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "project": "real-time-fraud-detection",
                "model_role": "deployable_champion",
                "feature_engineering": "missingness_indicators",
                "threshold": "0.60",
                "test_set_locked": "true",
                "preprocessing_included": "true",
                "deployment_ready": "true",
            }
        )

    print("\n" + "=" * 70)
    print("MLflow DEPLOYABLE MODEL CREATED")
    print("=" * 70)

    print("\nNext step:")
    print("Verify the newly registered model using MLflow.")
    print("=" * 70)


if __name__ == "__main__":
    main()