"""
Feature engineering based on concrete engineering domain knowledge.

Derived features include binder system ratios, aggregate features,
admixture intensity, and temporal transformations grounded in
hydration kinetics theory (ACI 211, IS 10262).
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw mix design and curing parameters.

    Adds 14 engineered features:
      Binder system (5): Binder, W_B_ratio, GGBS_ratio, FlyAsh_ratio, SCM_ratio
      Aggregate (3):     Total_Aggregate, Fine_Agg_ratio, Agg_Binder_ratio
      Admixture (1):     SP_per_binder
      Temporal (5):      log_Age, sqrt_Age, Age_very_early, Age_early, Age_standard

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the 8 raw input features.

    Returns
    -------
    pd.DataFrame
        DataFrame with original + derived features.
    """
    df_feat = df.copy()

    # ── Binder system features ──────────────────────────────────────────────
    # Total binder = cement + supplementary cementitious materials
    df_feat['Binder'] = (
        df_feat['Cement']
        + df_feat['Blast_Furnace_Slag']
        + df_feat['Fly_Ash']
    )

    # Water-to-binder ratio — single most important predictor (ACI 211)
    df_feat['W_B_ratio'] = df_feat['Water'] / df_feat['Binder']

    # SCM replacement ratios
    df_feat['GGBS_ratio'] = df_feat['Blast_Furnace_Slag'] / df_feat['Binder']
    df_feat['FlyAsh_ratio'] = df_feat['Fly_Ash'] / df_feat['Binder']
    df_feat['SCM_ratio'] = (
        (df_feat['Blast_Furnace_Slag'] + df_feat['Fly_Ash'])
        / df_feat['Binder']
    )

    # ── Aggregate features ──────────────────────────────────────────────────
    df_feat['Total_Aggregate'] = (
        df_feat['Coarse_Aggregate'] + df_feat['Fine_Aggregate']
    )
    df_feat['Fine_Agg_ratio'] = (
        df_feat['Fine_Aggregate'] / df_feat['Total_Aggregate']
    )
    df_feat['Agg_Binder_ratio'] = (
        df_feat['Total_Aggregate'] / df_feat['Binder']
    )

    # ── Admixture intensity ─────────────────────────────────────────────────
    df_feat['SP_per_binder'] = df_feat['Superplasticizer'] / df_feat['Binder']

    # ── Temporal transformations ────────────────────────────────────────────
    # Log-transform: strength develops log-linearly after ~7 days
    df_feat['log_Age'] = np.log1p(df_feat['Age'])

    # Square root: intermediate growth model
    df_feat['sqrt_Age'] = np.sqrt(df_feat['Age'])

    # Hydration phase indicators
    df_feat['Age_very_early'] = (df_feat['Age'] <= 3).astype(int)
    df_feat['Age_early'] = (
        (df_feat['Age'] > 3) & (df_feat['Age'] <= 7)
    ).astype(int)
    df_feat['Age_standard'] = (
        (df_feat['Age'] > 7) & (df_feat['Age'] <= 28)
    ).astype(int)

    # ── Sanity checks ──────────────────────────────────────────────────────
    # Replace any infinities from division by zero
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_nan = df_feat.isnull().sum().sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} NaN values created during feature engineering.")
        print("Filling NaN with 0 (likely from division by zero in ratio features).")
        df_feat.fillna(0, inplace=True)

    print(f"\nFeature engineering complete.")
    print(f"  Raw features:       8")
    print(f"  Engineered features: {len(df_feat.columns) - len(df.columns)}")
    print(f"  Total features:     {len(df_feat.columns) - 1} (excl. target)")

    return df_feat


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return all feature column names (excluding the target).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all features and target.

    Returns
    -------
    list
        Feature column names.
    """
    return [c for c in df.columns if c != 'Compressive_Strength']
