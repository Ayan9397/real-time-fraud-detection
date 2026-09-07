from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


SHAP_DIR = Path("models/shap")

SHAP_VALUES_PATH = SHAP_DIR / "shap_values.npy"
FEATURES_PATH = SHAP_DIR / "processed_features.npy"
NAMES_PATH = SHAP_DIR / "feature_names.csv"

OUTPUT_DIR = SHAP_DIR / "plots"


def main() -> None:
    print("Loading SHAP results...")

    shap_values = np.load(SHAP_VALUES_PATH)
    processed_features = np.load(FEATURES_PATH)

    feature_names = pd.read_csv(
        NAMES_PATH
    )["feature"].tolist()

    print(
        f"SHAP shape: {shap_values.shape}"
    )

    print(
        f"Feature matrix shape: {processed_features.shape}"
    )

    print(
        f"Feature names: {len(feature_names)}"
    )

    if shap_values.shape != processed_features.shape:
        raise ValueError(
            "SHAP values and feature matrix have different shapes."
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP values and feature names have different feature counts."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Global SHAP importance
    # ---------------------------------------------------------

    print("\nCreating global SHAP importance plot...")

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    top_n = 20

    top_indices = np.argsort(
        mean_abs_shap
    )[-top_n:]

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        range(top_n),
        mean_abs_shap[top_indices],
    )

    plt.yticks(
        range(top_n),
        [
            feature_names[i]
            for i in top_indices
        ],
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.title(
        "Top 20 Features by SHAP Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_shap_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved: global_shap_importance.png"
    )

    # ---------------------------------------------------------
    # 2. SHAP beeswarm plot
    # ---------------------------------------------------------

    print("\nCreating SHAP beeswarm plot...")

    shap.summary_plot(
        shap_values,
        processed_features,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved: shap_beeswarm.png"
    )

    # ---------------------------------------------------------
    # 3. SHAP bar summary using SHAP itself
    # ---------------------------------------------------------

    print(
        "\nCreating SHAP summary bar plot..."
    )

    shap.summary_plot(
        shap_values,
        processed_features,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary_bar.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved: shap_summary_bar.png"
    )

    # ---------------------------------------------------------
    # 4. Save top SHAP features as CSV
    # ---------------------------------------------------------

    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    shap_importance.head(50).to_csv(
        OUTPUT_DIR / "top_50_shap_features.csv",
        index=False,
    )

    print(
        "Saved: top_50_shap_features.csv"
    )

    # ---------------------------------------------------------
    # 5. Print top 20
    # ---------------------------------------------------------

    print("\nTop 20 SHAP features:")

    print(
        shap_importance.head(20).to_string(
            index=False
        )
    )

    print(
        "\nSHAP visualization completed successfully."
    )

    print(
        f"Plots saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()