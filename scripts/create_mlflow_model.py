import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_missing_model.joblib"

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "xgboost_missing_preprocessor.joblib"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "fraud-detection-modeling"

MODEL_NAME = "fraud_detection_xgboost"

MODEL_VERSION = "5"

THRESHOLD = 0.60


# ============================================================
# CUSTOM MLFLOW PYTHON MODEL
# ============================================================


class FraudDetectionModel(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for the missingness-aware XGBoost model.

    Training schema:
        432 original features
        + 432 missingness indicators
        = 864 features total

    The wrapper receives only the original transaction features
    and creates the missingness indicators automatically.
    """

    def load_context(self, context):

        self.model = joblib.load(context.artifacts["model"])

        self.preprocessor = joblib.load(context.artifacts["preprocessor"])

        self.expected_features = list(self.preprocessor.feature_names_in_)

        self.missing_features = [
            feature
            for feature in self.expected_features
            if feature.endswith("_missing")
        ]

        self.original_features = [
            feature
            for feature in self.expected_features
            if not feature.endswith("_missing")
        ]

        if len(self.original_features) != 432:
            raise RuntimeError(
                "Unexpected original feature count: "
                f"{len(self.original_features)}. "
                "Expected 432."
            )

        if len(self.missing_features) != 432:
            raise RuntimeError(
                "Unexpected missing-indicator count: "
                f"{len(self.missing_features)}. "
                "Expected 432."
            )

        if len(self.expected_features) != 864:
            raise RuntimeError(
                "Unexpected total feature count: "
                f"{len(self.expected_features)}. "
                "Expected 864."
            )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    def _prepare_input(self, model_input):

        if isinstance(model_input, dict):

            dataframe = pd.DataFrame([model_input])

        elif isinstance(model_input, pd.Series):

            dataframe = model_input.to_frame().T

        elif isinstance(model_input, pd.DataFrame):

            dataframe = model_input.copy()

        else:

            raise TypeError(
                "model_input must be a dictionary, "
                "pandas Series, or pandas DataFrame."
            )

        dataframe = dataframe.drop(
            columns=[
                "isFraud",
                "TransactionID",
            ],
            errors="ignore",
        )

        prepared_data = {
            feature: (
                dataframe[feature].iloc[0] if feature in dataframe.columns else np.nan
            )
            for feature in self.original_features
        }

        original_dataframe = pd.DataFrame(
            [prepared_data],
            columns=self.original_features,
        )

        missing_indicators = original_dataframe.isna().astype(np.int8)

        missing_indicators.columns = [
            f"{feature}_missing" for feature in self.original_features
        ]

        prepared_dataframe = pd.concat(
            [
                original_dataframe,
                missing_indicators,
            ],
            axis=1,
        )

        prepared_dataframe = prepared_dataframe[self.expected_features]

        return prepared_dataframe

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(
        self,
        context,
        model_input,
    ):

        start_time = time.perf_counter()

        dataframe = self._prepare_input(model_input)

        transformed = self.preprocessor.transform(dataframe)

        probabilities = self.model.predict_proba(transformed)[:, 1]

        predictions = probabilities >= THRESHOLD

        decisions = np.where(
            predictions,
            "FRAUD REVIEW",
            "LEGITIMATE",
        )

        inference_latency_ms = (time.perf_counter() - start_time) * 1000

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "fraud_prediction": predictions,
                "decision": decisions,
                "threshold": THRESHOLD,
                "prediction_latency_ms": (inference_latency_ms),
            }
        )


# ============================================================
# CREATE MLFLOW MODEL
# ============================================================


def main():

    print("=" * 70)
    print("CREATE PORTABLE MLFLOW FRAUD DETECTION MODEL")
    print("=" * 70)
    print()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():

        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow tracking URI: " f"{tracking_uri}")

    print()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:

        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

    else:

        experiment_id = experiment.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Using experiment: " f"{EXPERIMENT_NAME}")

    print(f"Experiment ID: " f"{experiment_id}")

    print()

    with mlflow.start_run(run_name="champion_xgboost_pyfunc_v5") as run:

        run_id = run.info.run_id

        print(f"MLflow Run ID: {run_id}")

        print()
        print("Logging portable MLflow model...")

        artifacts = {
            "model": str(MODEL_PATH.resolve()),
            "preprocessor": str(PREPROCESSOR_PATH.resolve()),
        }

        model_info = mlflow.pyfunc.log_model(
            name="fraud_detection_model",
            python_model=FraudDetectionModel(),
            artifacts=artifacts,
        )

        mlflow.log_param(
            "model_type",
            "XGBoost",
        )

        mlflow.log_param(
            "model_version",
            MODEL_VERSION,
        )

        mlflow.log_param(
            "threshold",
            THRESHOLD,
        )

        mlflow.log_param(
            "original_feature_count",
            432,
        )

        mlflow.log_param(
            "missing_indicator_count",
            432,
        )

        mlflow.log_param(
            "total_feature_count",
            864,
        )

        mlflow.set_tag(
            "model_role",
            "champion",
        )

        mlflow.set_tag(
            "deployment_stage",
            "production_candidate",
        )

        mlflow.set_tag(
            "architecture",
            "XGBoost + missingness-aware preprocessing",
        )

        mlflow.log_metrics(
            {
                "test_roc_auc": 0.8989,
                "test_pr_auc": 0.5203,
                "test_precision_at_060": 0.5522,
                "test_recall_at_060": 0.4583,
                "test_f1_at_060": 0.5009,
                "test_fpr_at_060": 0.013404,
            }
        )

        print()
        print("=" * 70)
        print("PORTABLE MLFLOW MODEL CREATED SUCCESSFULLY")
        print("=" * 70)
        print()
        print(f"Run ID: {run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Model ID: {model_info.model_id}")
        print(f"Model: {MODEL_NAME}")
        print("Original features: 432")
        print("Missing indicators: 432")
        print("Total features: 864")
        print(f"Threshold: {THRESHOLD}")
        print()
        print("MLflow model logging completed.")


if __name__ == "__main__":
    main()
