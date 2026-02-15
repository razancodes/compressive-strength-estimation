"""
SHAP-based explainability analysis for tree ensemble models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import shap


def shap_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    subset_name: str,
    output_dir: str = 'outputs',
) -> tuple:
    """
    Compute and visualize SHAP values for a trained tree ensemble model.

    Generates:
      1. Feature importance bar chart (global)
      2. Beeswarm summary plot (directional)
      3. Dependence plots for top-3 features

    Parameters
    ----------
    model : fitted model
        Trained tree model (XGBoost, CatBoost, or LightGBM).
    X_train : pd.DataFrame
        Training features (used as background for explainer).
    X_test : pd.DataFrame
        Test features to explain.
    model_name : str
        Model label for filenames.
    subset_name : str
        Subset label for filenames.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    tuple
        (shap_values ndarray, top_features list)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use a subsample for large datasets to keep SHAP computation fast
    max_samples = 500
    if len(X_test) > max_samples:
        X_explain = X_test.sample(n=max_samples, random_state=42)
    else:
        X_explain = X_test

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # 1. Global feature importance (bar chart)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name} ({subset_name})",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"shap_importance_{model_name}_{subset_name}.png"),
        dpi=300, bbox_inches='tight',
    )
    plt.close()

    # 2. Beeswarm summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title(f"SHAP Value Distribution — {model_name} ({subset_name})",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"shap_summary_{model_name}_{subset_name}.png"),
        dpi=300, bbox_inches='tight',
    )
    plt.close()

    # 3. Dependence plots for top-3 features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(feature_importance)[-3:][::-1]
    top_features = list(X_explain.columns[top_indices])

    for feature in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature, shap_values, X_explain, show=False)
        plt.title(f"SHAP Dependence — {feature}\n{model_name} ({subset_name})",
                  fontsize=13, fontweight='bold')
        plt.tight_layout()
        # Sanitize feature name for filename
        safe_feature = feature.replace('/', '_').replace(' ', '_')
        plt.savefig(
            os.path.join(output_dir,
                         f"shap_dep_{safe_feature}_{model_name}_{subset_name}.png"),
            dpi=300, bbox_inches='tight',
        )
        plt.close()

    print(f"  SHAP plots saved: importance, summary, + {len(top_features)} dependence plots")
    print(f"  Top-3 features: {top_features}")

    return shap_values, top_features
