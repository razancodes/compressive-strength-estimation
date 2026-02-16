#!/usr/bin/env python3
"""
Export trained models for inference.

Loads the optimized hyperparameters from training, retrains each model
on the full dataset, and saves model files in native + pickle formats.

Usage:
    python export_models.py

Output:
    models/
    ├── XGBoost_Full.json      # Native XGBoost
    ├── XGBoost_Full.pkl       # Pickle (universal)
    ├── CatBoost_Full.cbm      # Native CatBoost
    ├── CatBoost_Full.pkl
    ├── LightGBM_Full.txt      # Native LightGBM
    ├── LightGBM_Full.pkl
    ├── feature_config.json    # Feature engineering spec
    └── (same for EA1, EA7, EA14 subsets)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_and_preprocess_data, create_age_subsets
from src.feature_engineering import engineer_features, get_feature_columns

# ── Configuration ───────────────────────────────────────────────────────
DATA_PATH = 'data/Concrete_Data - Sheet1.csv'
PARAMS_PATH = 'outputs/best_hyperparameters.json'
OUTPUT_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data & engineer features ───────────────────────────────────────
print("Loading data and engineering features...")
df = load_and_preprocess_data(DATA_PATH)
df_feat = engineer_features(df)
feature_cols = get_feature_columns(df_feat)
subsets = create_age_subsets(df_feat)

# ── Load best hyperparameters from training ─────────────────────────────
print(f"\nLoading hyperparameters from {PARAMS_PATH}...")
with open(PARAMS_PATH, 'r') as f:
    all_best_params = json.load(f)

print(f"  Found {len(all_best_params)} model configurations:")
for key in sorted(all_best_params.keys()):
    print(f"    • {key}")

# ── Retrain & export each model ─────────────────────────────────────────
print(f"\nRetraining and saving models to {OUTPUT_DIR}/...\n")
saved = {}
t0 = time.time()

for key, params in all_best_params.items():
    # Parse key: "ModelType_SubsetName"
    parts = key.split('_', 1)
    model_type = parts[0]
    subset_name = parts[1]

    if subset_name not in subsets:
        print(f"  ⚠ Skipping {key} — subset '{subset_name}' not found")
        continue

    X = subsets[subset_name][feature_cols]
    y = subsets[subset_name]['Compressive_Strength']

    # Create and train model
    if model_type == 'XGBoost':
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X, y)
        native_ext = 'json'
        model.save_model(f"{OUTPUT_DIR}/{key}.{native_ext}")

    elif model_type == 'CatBoost':
        model = CatBoostRegressor(**params, random_state=42, verbose=0)
        model.fit(X, y)
        native_ext = 'cbm'
        model.save_model(f"{OUTPUT_DIR}/{key}.{native_ext}")

    elif model_type == 'LightGBM':
        model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X, y)
        native_ext = 'txt'
        model.booster_.save_model(f"{OUTPUT_DIR}/{key}.{native_ext}")

    else:
        print(f"  ⚠ Unknown model type: {model_type}")
        continue

    # Universal pickle
    joblib.dump(model, f"{OUTPUT_DIR}/{key}.pkl")

    native_size = os.path.getsize(f"{OUTPUT_DIR}/{key}.{native_ext}") / 1024
    pkl_size = os.path.getsize(f"{OUTPUT_DIR}/{key}.pkl") / 1024
    print(f"  ✅ {key:<25s}  .{native_ext} ({native_size:.0f} KB) + .pkl ({pkl_size:.0f} KB)")

    saved[key] = {
        'native': f"{key}.{native_ext}",
        'pkl': f"{key}.pkl",
    }

# ── Save feature configuration ──────────────────────────────────────────
feature_config = {
    'raw_features': [
        'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
        'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
    ],
    'all_features': feature_cols,
    'target': 'Compressive_Strength',
    'n_raw': 8,
    'n_engineered': len(feature_cols) - 8,
    'n_total': len(feature_cols),
    'subsets': list(subsets.keys()),
    'saved_models': saved,
    'best_hyperparameters': all_best_params,
}

config_path = f"{OUTPUT_DIR}/feature_config.json"
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)

# ── Summary ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  ✅ All models exported in {elapsed:.1f}s")
print(f"{'='*60}")
print(f"\n  Directory: {OUTPUT_DIR}/")
print(f"  Files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f"{OUTPUT_DIR}/{fname}")
    print(f"    {fname:<40s} {size/1024:.1f} KB")
print(f"\n  For inference, load any .pkl file with:")
print(f"    model = joblib.load('models/CatBoost_Full.pkl')")
print(f"    prediction = model.predict(X_features)")
