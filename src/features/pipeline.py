
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


class FraudFeaturePipeline:
    """Reusable preprocessing pipeline."""

    def __init__(self):
        self.pipeline = None

    def build(self, dataframe: pd.DataFrame):
        """Build preprocessing pipeline."""

        X = dataframe.drop(columns=["isFraud"])

        numeric_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        # Remove TransactionID because it's only an identifier.
        if "TransactionID" in numeric_features:
            numeric_features.remove("TransactionID")

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="missing",
                    ),
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    numeric_features,
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_features,
                ),
            ]
        )

        return self.pipeline

    def fit_transform(self, dataframe: pd.DataFrame):
        self.build(dataframe)

        X = dataframe.drop(columns=["isFraud"])

        return self.pipeline.fit_transform(X)

    def transform(self, dataframe: pd.DataFrame):
        X = dataframe.drop(columns=["isFraud"])

        return self.pipeline.transform(X)

    def save(self, path: str):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        self.pipeline = joblib.load(path)
        return self.pipeline