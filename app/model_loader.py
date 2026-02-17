"""
Shared inference module.

Handles model loading, feature engineering, prediction, and SHAP computation.
All Streamlit pages import from this module.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ── Paths ───────────────────────────────────────────────────────────────
# Resolve paths relative to the project root (parent of app/)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

# ── Input Field Definitions ────────────────────────────────────────────
INPUT_FIELDS = [
    {"name": "Cement",            "unit": "kg/m3", "min": 100.0, "max": 550.0, "default": 280.0, "step": 5.0},
    {"name": "Blast_Furnace_Slag", "unit": "kg/m3", "min": 0.0,   "max": 360.0, "default": 50.0,  "step": 5.0},
    {"name": "Fly_Ash",           "unit": "kg/m3", "min": 0.0,   "max": 200.0, "default": 50.0,  "step": 5.0},
    {"name": "Water",             "unit": "kg/m3", "min": 120.0, "max": 250.0, "default": 180.0, "step": 5.0},
    {"name": "Superplasticizer",  "unit": "kg/m3", "min": 0.0,   "max": 32.0,  "default": 6.0,   "step": 0.5},
    {"name": "Coarse_Aggregate",  "unit": "kg/m3", "min": 800.0, "max": 1150.0,"default": 970.0, "step": 5.0},
    {"name": "Fine_Aggregate",    "unit": "kg/m3", "min": 590.0, "max": 950.0, "default": 770.0, "step": 5.0},
    {"name": "Age",               "unit": "days",  "min": 1.0,   "max": 365.0, "default": 28.0,  "step": 1.0},
]

# Display-friendly names for UI labels
DISPLAY_NAMES = {
    "Cement": "Cement",
    "Blast_Furnace_Slag": "Blast Furnace Slag",
    "Fly_Ash": "Fly Ash",
    "Water": "Water",
    "Superplasticizer": "Superplasticizer",
    "Coarse_Aggregate": "Coarse Aggregate",
    "Fine_Aggregate": "Fine Aggregate",
    "Age": "Age",
}

MODEL_TYPES = ["XGBoost", "CatBoost", "LightGBM"]
SUBSET_NAMES = ["EA1", "EA7", "EA14", "Full"]

# Best-practice defaults for missing values.
# SCMs and SP default to 0 (not all mixes use them).
# Age defaults to 28 days (standard test age per ASTM C39).
# Other values are training-data medians rounded to practical values.
DEFAULT_FILL_VALUES = {
    "Cement": 280.0,              # Median cement content (kg/m3)
    "Blast_Furnace_Slag": 0.0,    # Not all mixes use GGBS
    "Fly_Ash": 0.0,               # Not all mixes use fly ash
    "Water": 180.0,               # Typical water content (kg/m3)
    "Superplasticizer": 0.0,      # Not all mixes use SP
    "Coarse_Aggregate": 968.0,    # Training median (kg/m3)
    "Fine_Aggregate": 774.0,      # Training median (kg/m3)
    "Age": 28.0,                  # Standard 28-day test (ASTM C39)
}


def autofill_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Fill missing/null values with engineering best-practice defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Input data that may contain NaN values.

    Returns
    -------
    tuple of (pd.DataFrame, list[str])
        Filled DataFrame and list of human-readable fill messages.
    """
    df = df.copy()
    fill_log = []

    for col, default in DEFAULT_FILL_VALUES.items():
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(default)
                fill_log.append(
                    f"{col}: {n_missing} missing values filled with "
                    f"{default} ({'0 = not used' if default == 0 else 'training median'})"
                )

    return df, fill_log


# ── Data Loading (cached) ──────────────────────────────────────────────

@st.cache_data
def load_feature_config():
    """Load feature_config.json from the models directory."""
    path = os.path.join(MODELS_DIR, "feature_config.json")
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_results_df():
    """Load the training results summary CSV."""
    path = os.path.join(OUTPUTS_DIR, "stage_a_results_summary.csv")
    return pd.read_csv(path)


@st.cache_data
def load_hyperparameters():
    """Load best hyperparameters JSON."""
    path = os.path.join(OUTPUTS_DIR, "best_hyperparameters.json")
    with open(path, "r") as f:
        return json.load(f)


# ── Model Loading (cached) ─────────────────────────────────────────────

@st.cache_resource
def load_model(model_type: str, subset: str):
    """
    Load a trained model from pickle file.

    Parameters
    ----------
    model_type : str
        One of 'XGBoost', 'CatBoost', 'LightGBM'.
    subset : str
        One of 'EA1', 'EA7', 'EA14', 'Full'.

    Returns
    -------
    Trained model object.
    """
    # Whitelist validation — prevent path traversal
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. "
                         f"Must be one of {MODEL_TYPES}")
    if subset not in SUBSET_NAMES:
        raise ValueError(f"Invalid subset: {subset}. "
                         f"Must be one of {SUBSET_NAMES}")

    key = f"{model_type}_{subset}"
    path = os.path.join(MODELS_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ── Feature Engineering ────────────────────────────────────────────────

def engineer_features(raw_input: dict) -> pd.DataFrame:
    """
    Apply the 14 derived features to a raw 8-value input.

    Parameters
    ----------
    raw_input : dict
        Keys: Cement, Blast_Furnace_Slag, Fly_Ash, Water,
              Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 22 features (8 raw + 14 engineered).
    """
    df = pd.DataFrame([raw_input])

    # Binder system
    df["Binder"] = df["Cement"] + df["Blast_Furnace_Slag"] + df["Fly_Ash"]
    df["W_B_ratio"] = df["Water"] / df["Binder"]
    df["GGBS_ratio"] = df["Blast_Furnace_Slag"] / df["Binder"]
    df["FlyAsh_ratio"] = df["Fly_Ash"] / df["Binder"]
    df["SCM_ratio"] = (df["Blast_Furnace_Slag"] + df["Fly_Ash"]) / df["Binder"]

    # Aggregate
    df["Total_Aggregate"] = df["Coarse_Aggregate"] + df["Fine_Aggregate"]
    df["Fine_Agg_ratio"] = df["Fine_Aggregate"] / df["Total_Aggregate"]
    df["Agg_Binder_ratio"] = df["Total_Aggregate"] / df["Binder"]

    # Admixture
    df["SP_per_binder"] = df["Superplasticizer"] / df["Binder"]

    # Temporal
    df["log_Age"] = np.log1p(df["Age"])
    df["sqrt_Age"] = np.sqrt(df["Age"])
    df["Age_very_early"] = (df["Age"] <= 3).astype(int)
    df["Age_early"] = ((df["Age"] > 3) & (df["Age"] <= 7)).astype(int)
    df["Age_standard"] = ((df["Age"] > 7) & (df["Age"] <= 28)).astype(int)

    # Handle edge cases
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def engineer_features_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to a multi-row DataFrame.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame with the 8 raw columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 22 feature columns.
    """
    df = df_raw.copy()

    df["Binder"] = df["Cement"] + df["Blast_Furnace_Slag"] + df["Fly_Ash"]
    df["W_B_ratio"] = df["Water"] / df["Binder"]
    df["GGBS_ratio"] = df["Blast_Furnace_Slag"] / df["Binder"]
    df["FlyAsh_ratio"] = df["Fly_Ash"] / df["Binder"]
    df["SCM_ratio"] = (df["Blast_Furnace_Slag"] + df["Fly_Ash"]) / df["Binder"]
    df["Total_Aggregate"] = df["Coarse_Aggregate"] + df["Fine_Aggregate"]
    df["Fine_Agg_ratio"] = df["Fine_Aggregate"] / df["Total_Aggregate"]
    df["Agg_Binder_ratio"] = df["Total_Aggregate"] / df["Binder"]
    df["SP_per_binder"] = df["Superplasticizer"] / df["Binder"]
    df["log_Age"] = np.log1p(df["Age"])
    df["sqrt_Age"] = np.sqrt(df["Age"])
    df["Age_very_early"] = (df["Age"] <= 3).astype(int)
    df["Age_early"] = ((df["Age"] > 3) & (df["Age"] <= 7)).astype(int)
    df["Age_standard"] = ((df["Age"] > 7) & (df["Age"] <= 28)).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# ── Prediction ──────────────────────────────────────────────────────────

def get_feature_columns():
    """Return the ordered list of 22 feature columns for model input."""
    config = load_feature_config()
    return config["all_features"]


def predict(model, features_df: pd.DataFrame) -> float:
    """
    Run model prediction on engineered features.

    Parameters
    ----------
    model : trained model
    features_df : pd.DataFrame
        One-row DataFrame with 22 feature columns.

    Returns
    -------
    float
        Predicted compressive strength in MPa.
    """
    cols = get_feature_columns()
    X = features_df[cols]
    return float(model.predict(X)[0])


def predict_batch(model, features_df: pd.DataFrame) -> np.ndarray:
    """Predict on a multi-row DataFrame. Returns array of predictions."""
    cols = get_feature_columns()
    X = features_df[cols]
    return model.predict(X)


# ── SHAP ────────────────────────────────────────────────────────────────

def get_shap_explanation(model, features_df: pd.DataFrame):
    """
    Compute SHAP values for a single prediction.

    Returns a shap.Explanation object suitable for waterfall/force plots.
    """
    cols = get_feature_columns()
    X = features_df[cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X.values[0],
        feature_names=cols,
    )


# ── Subset Selection Logic ─────────────────────────────────────────────

def select_subset_for_age(age: float) -> str:
    """
    Select the most appropriate model subset for a given age.

    Uses the most specialized model available:
    - Age <= 3  -> EA1
    - Age <= 7  -> EA7
    - Age <= 14 -> EA14
    - Age > 14  -> Full
    """
    if age <= 3:
        return "EA1"
    elif age <= 7:
        return "EA7"
    elif age <= 14:
        return "EA14"
    else:
        return "Full"


# ── Plot Paths ──────────────────────────────────────────────────────────

def get_plot_path(plot_type: str, model_type: str = None, subset: str = None,
                  feature: str = None) -> str:
    """
    Build the path to a pre-computed plot image.

    Parameters
    ----------
    plot_type : str
        One of: 'pred_scatter', 'residuals', 'shap_importance',
        'shap_summary', 'shap_dep', 'eda_target_dist',
        'eda_strength_vs_age', 'eda_correlation', 'r2_heatmap',
        'model_comparison'.
    model_type : str, optional
    subset : str, optional
    feature : str, optional
        For shap_dep plots.

    Returns
    -------
    str
        Absolute path to the image file.
    """
    if plot_type in ("eda_target_dist", "eda_strength_vs_age", "eda_correlation",
                      "r2_heatmap", "model_comparison"):
        filename = f"{plot_type}.png"
    elif plot_type == "shap_dep":
        filename = f"shap_dep_{feature}_{model_type}_{subset}.png"
    else:
        filename = f"{plot_type}_{model_type}_{subset}.png"

    return os.path.join(OUTPUTS_DIR, filename)


def plot_exists(path: str) -> bool:
    """Check if a plot file exists."""
    return os.path.isfile(path)


# ── Input Form Component ───────────────────────────────────────────────

def render_input_form(include_age: bool = True, key_prefix: str = ""):
    """
    Render the 8-input form using Streamlit number_input widgets.

    Parameters
    ----------
    include_age : bool
        Whether to include the Age field (excluded for strength curve).
    key_prefix : str
        Prefix for widget keys to avoid conflicts across pages.

    Returns
    -------
    dict
        Raw input values.
    """
    raw_input = {}

    for field in INPUT_FIELDS:
        name = field["name"]
        if name == "Age" and not include_age:
            continue

        display = DISPLAY_NAMES[name]
        label = f"{display} ({field['unit']})"

        val = st.number_input(
            label,
            min_value=field["min"],
            max_value=field["max"],
            value=field["default"],
            step=field["step"],
            key=f"{key_prefix}{name}",
        )
        raw_input[name] = val

    return raw_input
