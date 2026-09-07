from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "fraud-detection-modeling"
RUN_NAME = "xgboost-missingness-champion"
REGISTERED_MODEL_NAME = "fraud_detection_xgboost"


def main() -> None:

    print("Connecting to MLflow...")

    tracking_uri = (
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    client = MlflowClient()

    # ---------------------------------------------------------
    # 1. Find the experiment
    # ---------------------------------------------------------

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    print(
        f"Experiment ID: {experiment.experiment_id}"
    )

    # ---------------------------------------------------------
    # 2. Find the champion run
    # ---------------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            f"tags.mlflow.runName = '{RUN_NAME}'"
        ),
        order_by=[
            "attributes.start_time DESC"
        ],
    )

    if not runs:
        raise ValueError(
            f"Run not found: {RUN_NAME}"
        )

    run = runs[0]

    run_id = run.info.run_id

    print(
        f"Champion run ID: {run_id}"
    )

    # ---------------------------------------------------------
    # 3. Check the model artifact
    # ---------------------------------------------------------

    model_artifact = (
        "model_files/xgboost_missing_model.joblib"
    )

    print(
        f"Model artifact: {model_artifact}"
    )

    # ---------------------------------------------------------
    # 4. Create registered model if necessary
    # ---------------------------------------------------------

    try:

        registered_model = (
            client.get_registered_model(
                REGISTERED_MODEL_NAME
            )
        )

        print(
            f"Registered model already exists: "
            f"{registered_model.name}"
        )

    except Exception:

        print(
            "Registered model does not exist."
        )

        print(
            "Creating registered model..."
        )

        registered_model = (
            client.create_registered_model(
                REGISTERED_MODEL_NAME
            )
        )

        print(
            f"Created: {registered_model.name}"
        )

    # ---------------------------------------------------------
    # 5. Create model version
    # ---------------------------------------------------------

    print(
        "\nCreating model version..."
    )

    model_version = (
        client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model_artifact,
            run_id=run_id,
        )
    )

    print(
        f"Model version: {model_version.version}"
    )

    print(
        f"Model status: {model_version.status}"
    )

    # ---------------------------------------------------------
    # 6. Add model version description
    # ---------------------------------------------------------

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=(
            "Champion XGBoost fraud detection model "
            "trained using the IEEE-CIS Fraud Detection "
            "dataset. Uses original features plus "
            "missingness indicators. Selected using "
            "validation PR-AUC with threshold 0.60."
        ),
    )

    # ---------------------------------------------------------
    # 7. Add model tags
    # ---------------------------------------------------------

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="model_role",
        value="champion",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="threshold",
        value="0.60",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="feature_strategy",
        value="missingness_indicators",
    )

    # ---------------------------------------------------------
    # 8. Print final information
    # ---------------------------------------------------------

    print(
        "\nModel registration completed successfully."
    )

    print(
        f"Registered model: {REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version: {model_version.version}"
    )

    print(
        f"Source run: {run_id}"
    )


if __name__ == "__main__":
    main()