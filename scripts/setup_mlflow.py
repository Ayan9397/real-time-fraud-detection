from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

CHAMPION_MODEL = (
    MODEL_DIR / "xgboost_missing_model.joblib"
)

CHAMPION_PREPROCESSOR = (
    MODEL_DIR / "xgboost_missing_preprocessor.joblib"
)

EXPERIMENT_NAME = "fraud-detection-modeling"


def main() -> None:

    print("Setting up MLflow...")

    # ---------------------------------------------------------
    # 1. Configure SQLite MLflow tracking
    # ---------------------------------------------------------

    tracking_uri = (
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    print(
        f"Tracking URI: {tracking_uri}"
    )

    # ---------------------------------------------------------
    # 2. Create / select experiment
    # ---------------------------------------------------------

    experiment = mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"Experiment: {experiment.name}"
    )

    print(
        f"Experiment ID: {experiment.experiment_id}"
    )

    # ---------------------------------------------------------
    # 3. Start champion model run
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name="xgboost-missingness-champion"
    ):

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        print(
            f"Run ID: {run_id}"
        )

        # -----------------------------------------------------
        # 4. Log model parameters
        # -----------------------------------------------------

        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "feature_strategy": (
                    "original_features_plus_missing_indicators"
                ),
                "original_features": 432,
                "missing_indicators": 432,
                "total_features_before_encoding": 864,
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "tree_method": "hist",
                "eval_metric": "aucpr",
                "class_weight_strategy": (
                    "scale_pos_weight"
                ),
                "selected_threshold": 0.60,
                "validation_sample_for_shap": 5000,
            }
        )

        # -----------------------------------------------------
        # 5. Log validation metrics
        # -----------------------------------------------------

        mlflow.log_metrics(
            {
                "validation_precision": 0.5185,
                "validation_recall": 0.5664,
                "validation_f1": 0.5414,
                "validation_roc_auc": 0.9197,
                "validation_pr_auc": 0.5827,
            }
        )

        # -----------------------------------------------------
        # 6. Log final locked test metrics
        #
        # These metrics are recorded for final evaluation only.
        # They were NOT used for model selection or threshold
        # optimization.
        # -----------------------------------------------------

        mlflow.log_metrics(
            {
                "test_precision_at_0_60": 0.5522,
                "test_recall_at_0_60": 0.4583,
                "test_f1_at_0_60": 0.5009,
                "test_roc_auc": 0.8989,
                "test_pr_auc": 0.5203,
            }
        )

        # -----------------------------------------------------
        # 7. Log project tags
        # -----------------------------------------------------

        mlflow.set_tags(
            {
                "project": (
                    "real-time-fraud-detection"
                ),
                "dataset": (
                    "IEEE-CIS Fraud Detection"
                ),
                "split_strategy": "temporal",
                "model_status": "champion",
                "explainability": "SHAP",
                "test_set_locked": "true",
                "threshold_selected_on": (
                    "validation"
                ),
                "feature_engineering": (
                    "missingness_indicators"
                ),
            }
        )

        # -----------------------------------------------------
        # 8. Log champion model artifact
        # -----------------------------------------------------

        print(
            "\nLogging champion model..."
        )

        mlflow.log_artifact(
            str(CHAMPION_MODEL),
            artifact_path="model_files",
        )

        # -----------------------------------------------------
        # 9. Log preprocessor artifact
        # -----------------------------------------------------

        print(
            "Logging preprocessor..."
        )

        mlflow.log_artifact(
            str(CHAMPION_PREPROCESSOR),
            artifact_path="model_files",
        )

        # -----------------------------------------------------
        # 10. Log SHAP artifacts
        # -----------------------------------------------------

        shap_dir = (
            MODEL_DIR / "shap"
        )

        if shap_dir.exists():

            print(
                "Logging SHAP artifacts..."
            )

            for file_path in shap_dir.iterdir():

                if file_path.is_file():

                    mlflow.log_artifact(
                        str(file_path),
                        artifact_path="shap",
                    )

            plots_dir = (
                shap_dir / "plots"
            )

            if plots_dir.exists():

                mlflow.log_artifacts(
                    str(plots_dir),
                    artifact_path="shap/plots",
                )

        # -----------------------------------------------------
        # 11. Load champion model metadata
        # -----------------------------------------------------

        model = joblib.load(
            CHAMPION_MODEL
        )

        # -----------------------------------------------------
        # 12. Create metadata file
        # -----------------------------------------------------

        metadata_path = (
            MODEL_DIR
            / "champion_metadata.txt"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "Fraud Detection Champion Model\n"
            )

            file.write(
                "================================\n\n"
            )

            file.write(
                "Model: XGBoost\n"
            )

            file.write(
                "Feature strategy: "
                "Original + missing indicators\n"
            )

            file.write(
                "Original features: 432\n"
            )

            file.write(
                "Missing indicators: 432\n"
            )

            file.write(
                "Total features before encoding: 864\n"
            )

            file.write(
                "Validation PR-AUC: 0.5827\n"
            )

            file.write(
                "Validation ROC-AUC: 0.9197\n"
            )

            file.write(
                "Validation F1: 0.5414\n"
            )

            file.write(
                "Selected threshold: 0.60\n"
            )

            file.write(
                "Test PR-AUC: 0.5203\n"
            )

            file.write(
                "Test ROC-AUC: 0.8989\n"
            )

            file.write(
                "Test F1 at threshold 0.60: 0.5009\n"
            )

            file.write(
                "\nModel object type:\n"
            )

            file.write(
                f"{type(model)}\n"
            )

            file.write(
                f"\nMLflow run ID:\n{run_id}\n"
            )

        # -----------------------------------------------------
        # 13. Log metadata
        # -----------------------------------------------------

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="metadata",
        )

        # -----------------------------------------------------
        # 14. Completion
        # -----------------------------------------------------

        print(
            "\nMLflow run completed."
        )

        print(
            f"Run ID: {run_id}"
        )

    print(
        "\nMLflow setup completed successfully."
    )

    print(
        f"MLflow database: {MLFLOW_DB}"
    )


if __name__ == "__main__":
    main()