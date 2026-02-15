"""
Generates Stage_A_Training.ipynb — a self-contained Colab notebook
for training concrete strength prediction models with GPU support.

Run: python generate_notebook.py
"""

import json

cells = []


def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    })


def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "execution_count": None,
        "outputs": [],
    })


# =========================================================================
# CELL 1: Title
# =========================================================================
md("""\
# Stage A: Early-Age Concrete Compressive Strength Prediction
## ML Baseline Training Notebook (Colab GPU)

**Project:** Data-Driven Early-Age Concrete Strength Prediction
**Phase:** Stage A — Machine Learning Baseline on UCI Concrete Dataset

**Models:** XGBoost · CatBoost · LightGBM (Optuna-optimized)
**Validation:** 5-fold Nested Cross-Validation (unbiased estimates)
**Explainability:** SHAP TreeExplainer analysis

---
### How to use this notebook:
1. Set **Runtime → Change runtime type → T4 GPU** (recommended but optional)
2. Run cells sequentially from top to bottom
3. Upload your `Concrete_Data` CSV when prompted
4. After training completes, download the `models/` ZIP from the final cell

> **Quick test:** Set `N_TRIALS = 10` in the Configuration cell for a ~15 min run
> **Full run:** Set `N_TRIALS = 100` for publication-grade results (~2-4 hours on GPU)
""")

# =========================================================================
# CELL 2: Installation
# =========================================================================
md("## 1. Install Dependencies")

code("""\
!pip install -q xgboost catboost lightgbm optuna shap --upgrade
print("\\n✅ All packages installed successfully!")
""")

# =========================================================================
# CELL 3: Imports + Configuration
# =========================================================================
md("## 2. Imports & Configuration")

code("""\
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import warnings
import shutil
import subprocess

from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import shap
import joblib

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION — Adjust these values before running
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_TRIALS = 100          # Optuna trials per model-subset (10 for quick test)
N_OUTER_FOLDS = 5       # Outer CV folds for nested cross-validation
RANDOM_SEED = 42        # Global seed for reproducibility
SUBSET_NAMES = ['EA1', 'EA7', 'EA14', 'Full']  # Remove any to skip

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# GPU detection
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    GPU_AVAILABLE = result.returncode == 0
except FileNotFoundError:
    GPU_AVAILABLE = False

print(f"GPU Available: {GPU_AVAILABLE}")
print(f"Optuna trials per model-subset: {N_TRIALS}")
print(f"Subsets to evaluate: {SUBSET_NAMES}")

if GPU_AVAILABLE:
    !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print("\\n✅ GPU acceleration will be used for XGBoost and CatBoost")
else:
    print("\\n⚠️  No GPU detected. Training will use CPU (slower).")
    print("    Set Runtime → Change runtime type → T4 GPU for speedup.")

print(f"\\nxgboost={xgb.__version__}, catboost={CatBoostRegressor.__module__}, lightgbm={lgb.__version__}")
print(f"optuna={optuna.__version__}, shap={shap.__version__}")
""")

# =========================================================================
# CELL 4: Data Upload & Preprocessing
# =========================================================================
md("""\
## 3. Data Loading & Preprocessing

Upload the **Concrete_Data CSV** file when prompted.
Expected: 1030 rows × 9 columns (8 inputs + 1 target).
""")

code("""\
from google.colab import files

print("📁 Upload your Concrete_Data CSV file:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"\\nUploaded: {filename}")

# Load data
df_raw = pd.read_csv(filename)

# Standardize column names
EXPECTED_COLS = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate',
    'Age', 'Compressive_Strength'
]

# Try to map known long column names to short names
COLUMN_MAP = {
    'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast_Furnace_Slag',
    'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly_Ash',
    'Water  (component 4)(kg in a m^3 mixture)': 'Water',
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse_Aggregate',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine_Aggregate',
    'Age (day)': 'Age',
}
# Handle target column (may have trailing space)
for col in df_raw.columns:
    stripped = col.strip()
    if 'compressive' in stripped.lower() and 'strength' in stripped.lower():
        COLUMN_MAP[col] = 'Compressive_Strength'

rename_dict = {}
for col in df_raw.columns:
    stripped = col.strip()
    if col in COLUMN_MAP:
        rename_dict[col] = COLUMN_MAP[col]
    elif stripped in COLUMN_MAP:
        rename_dict[col] = COLUMN_MAP[stripped]

if rename_dict:
    df_raw = df_raw.rename(columns=rename_dict)

# Fallback: if columns still don't match, assign by position
if not all(c in df_raw.columns for c in EXPECTED_COLS):
    if len(df_raw.columns) == 9:
        print("⚠️  Column names not recognized. Assigning by position.")
        df_raw.columns = EXPECTED_COLS
    else:
        raise ValueError(f"Expected 9 columns, got {len(df_raw.columns)}: {list(df_raw.columns)}")

# Clean
df_raw = df_raw.drop_duplicates()
assert df_raw.isnull().sum().sum() == 0, f"Found missing values:\\n{df_raw.isnull().sum()}"

print(f"\\n✅ Dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"\\nTarget statistics (Compressive Strength, MPa):")
print(df_raw['Compressive_Strength'].describe().round(2))
print(f"\\nAge values: {sorted(df_raw['Age'].unique())}")

# Save a clean copy for reference
df_raw.to_csv('data_clean.csv', index=False)
df_raw.head(8)
""")

# =========================================================================
# CELL 5: Feature Engineering
# =========================================================================
md("""\
## 4. Feature Engineering

14 domain-derived features based on concrete engineering theory:
- **Binder system (5):** Binder, W/B ratio, GGBS ratio, Fly Ash ratio, SCM ratio
- **Aggregate (3):** Total Aggregate, Fine Agg ratio, Agg/Binder ratio
- **Admixture (1):** SP per binder
- **Temporal (5):** log(Age), √Age, age phase indicators
""")

code("""\
def engineer_features(df):
    \"\"\"Create 14 derived features from raw mix design parameters.\"\"\"
    d = df.copy()

    # Binder system
    d['Binder'] = d['Cement'] + d['Blast_Furnace_Slag'] + d['Fly_Ash']
    d['W_B_ratio'] = d['Water'] / d['Binder']
    d['GGBS_ratio'] = d['Blast_Furnace_Slag'] / d['Binder']
    d['FlyAsh_ratio'] = d['Fly_Ash'] / d['Binder']
    d['SCM_ratio'] = (d['Blast_Furnace_Slag'] + d['Fly_Ash']) / d['Binder']

    # Aggregate
    d['Total_Aggregate'] = d['Coarse_Aggregate'] + d['Fine_Aggregate']
    d['Fine_Agg_ratio'] = d['Fine_Aggregate'] / d['Total_Aggregate']
    d['Agg_Binder_ratio'] = d['Total_Aggregate'] / d['Binder']

    # Admixture
    d['SP_per_binder'] = d['Superplasticizer'] / d['Binder']

    # Temporal
    d['log_Age'] = np.log1p(d['Age'])
    d['sqrt_Age'] = np.sqrt(d['Age'])
    d['Age_very_early'] = (d['Age'] <= 3).astype(int)
    d['Age_early'] = ((d['Age'] > 3) & (d['Age'] <= 7)).astype(int)
    d['Age_standard'] = ((d['Age'] > 7) & (d['Age'] <= 28)).astype(int)

    # Handle inf/nan from division by zero
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.fillna(0, inplace=True)

    return d

df_feat = engineer_features(df_raw)
feature_cols = [c for c in df_feat.columns if c != 'Compressive_Strength']
print(f"✅ {len(feature_cols)} features: {feature_cols}")
""")

# =========================================================================
# CELL 6: EDA
# =========================================================================
md("## 5. Exploratory Data Analysis")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Target distribution
axes[0].hist(df_raw['Compressive_Strength'], bins=40, edgecolor='white', color='#2196F3', alpha=0.85)
axes[0].set_xlabel('Compressive Strength (MPa)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution', fontweight='bold')
axes[0].grid(alpha=0.2)

# Strength vs Age
axes[1].scatter(df_raw['Age'], df_raw['Compressive_Strength'], alpha=0.4, s=25, c='#FF5722', edgecolors='white', linewidths=0.3)
axes[1].set_xlabel('Age (days)')
axes[1].set_ylabel('Compressive Strength (MPa)')
axes[1].set_title('Strength vs Curing Age', fontweight='bold')
axes[1].grid(alpha=0.2)

# W/B ratio vs Strength
axes[2].scatter(df_feat['W_B_ratio'], df_raw['Compressive_Strength'], alpha=0.4, s=25, c='#4CAF50', edgecolors='white', linewidths=0.3)
axes[2].set_xlabel('Water-to-Binder Ratio')
axes[2].set_ylabel('Compressive Strength (MPa)')
axes[2].set_title('Strength vs W/B Ratio', fontweight='bold')
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/eda_overview.png', dpi=200, bbox_inches='tight')
plt.show()

# Correlation heatmap
raw_cols = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
            'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate',
            'Age', 'Compressive_Strength']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_raw[raw_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'fontsize': 9})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/eda_correlation.png', dpi=200, bbox_inches='tight')
plt.show()
""")

# =========================================================================
# CELL 7: Age Subsets
# =========================================================================
md("""\
## 6. Early-Age Subset Creation

| Subset | Age Range | Purpose |
|--------|-----------|---------|
| EA1    | ≤ 3 days  | Formwork removal decisions |
| EA7    | ≤ 7 days  | Construction sequencing |
| EA14   | ≤ 14 days | Quality assurance |
| Full   | All ages  | Baseline comparison |
""")

code("""\
ALL_SUBSETS = {
    'EA1':  df_feat[df_feat['Age'] <= 3].copy(),
    'EA7':  df_feat[df_feat['Age'] <= 7].copy(),
    'EA14': df_feat[df_feat['Age'] <= 14].copy(),
    'Full': df_feat.copy(),
}

# Filter to only requested subsets
subsets = {k: v for k, v in ALL_SUBSETS.items() if k in SUBSET_NAMES}

print(f"{'Subset':<8} {'Age Range':<12} {'Samples':>8}")
print("-" * 32)
for name, sdf in subsets.items():
    age_range = f"1-{int(sdf['Age'].max())}d"
    print(f"{name:<8} {age_range:<12} {len(sdf):>8}")
""")

# =========================================================================
# CELL 8: Helper Functions
# =========================================================================
md("""\
## 7. Model Training Functions

Defines:
- Manual K-fold CV (CatBoost/sklearn compatibility)
- Optuna objectives for XGBoost, CatBoost, LightGBM
- Nested cross-validation pipeline
- Model factory
""")

code("""\
# ── Manual CV (for CatBoost sklearn compatibility) ──────────────────────

def manual_cv_rmse(model_class, params, X, y, n_splits=5):
    \"\"\"Manual KFold CV — avoids CatBoost/sklearn __sklearn_tags__ issue.\"\"\"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    X_np = X.values if hasattr(X, 'values') else np.array(X)
    y_np = y.values if hasattr(y, 'values') else np.array(y)

    for train_idx, val_idx in kf.split(X_np):
        model = model_class(**params)
        model.fit(X_np[train_idx], y_np[train_idx])
        y_pred = model.predict(X_np[val_idx])
        scores.append(np.sqrt(mean_squared_error(y_np[val_idx], y_pred)))

    return np.mean(scores)


# ── Optuna Objectives ───────────────────────────────────────────────────

def objective_xgboost(trial, X, y):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': RANDOM_SEED, 'n_jobs': -1,
    }
    if GPU_AVAILABLE:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()


def objective_catboost(trial, X, y):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
        'random_state': RANDOM_SEED, 'verbose': 0,
    }
    if GPU_AVAILABLE:
        params['task_type'] = 'GPU'
        params['devices'] = '0'
    return manual_cv_rmse(CatBoostRegressor, params, X, y)


def objective_lightgbm(trial, X, y):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()


OBJECTIVE_MAP = {
    'XGBoost':  objective_xgboost,
    'CatBoost': objective_catboost,
    'LightGBM': objective_lightgbm,
}


# ── Model Factory ──────────────────────────────────────────────────────

def create_model(model_type, params):
    \"\"\"Instantiate a model with best params + GPU config.\"\"\"
    p = dict(params)
    if model_type == 'XGBoost':
        if GPU_AVAILABLE:
            p['tree_method'] = 'hist'
            p['device'] = 'cuda'
        return xgb.XGBRegressor(**p, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_type == 'CatBoost':
        if GPU_AVAILABLE:
            p['task_type'] = 'GPU'
            p['devices'] = '0'
        return CatBoostRegressor(**p, random_state=RANDOM_SEED, verbose=0)
    elif model_type == 'LightGBM':
        return lgb.LGBMRegressor(**p, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {model_type}")


# ── Nested Cross-Validation ────────────────────────────────────────────

def nested_cv(X, y, model_type, n_outer=N_OUTER_FOLDS, n_trials=N_TRIALS):
    \"\"\"Full nested CV: outer folds for evaluation, inner Optuna for HPO.\"\"\"
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_SEED)
    objective_fn = OBJECTIVE_MAP[model_type]

    metrics = {'rmse': [], 'mae': [], 'r2': [], 'mape': [], 'max_err': []}
    y_true_all, y_pred_all = [], []
    best_params_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"    Fold {fold+1}/{n_outer}", end=" ... ", flush=True)
        t0 = time.time()

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        # Inner HPO
        study = optuna.create_study(direction='minimize',
                                    sampler=TPESampler(seed=RANDOM_SEED + fold))
        study.optimize(lambda trial: objective_fn(trial, X_tr, y_tr),
                       n_trials=n_trials, show_progress_bar=False)

        best_p = study.best_params
        best_params_per_fold.append(best_p)

        # Train on outer train, predict on outer test
        model = create_model(model_type, best_p)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        mask = y_te != 0
        mape = np.mean(np.abs((y_te[mask] - y_pred[mask.values]) / y_te[mask])) * 100 if mask.sum() > 0 else np.nan
        max_e = np.max(np.abs(y_te.values - y_pred))

        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        metrics['mape'].append(mape)
        metrics['max_err'].append(max_e)
        y_true_all.extend(y_te.values.tolist())
        y_pred_all.extend(y_pred.tolist())

        print(f"RMSE={rmse:.2f}, R²={r2:.4f} ({time.time()-t0:.0f}s)")

    return {
        'RMSE_mean': np.mean(metrics['rmse']),  'RMSE_std': np.std(metrics['rmse']),
        'MAE_mean':  np.mean(metrics['mae']),   'MAE_std':  np.std(metrics['mae']),
        'R2_mean':   np.mean(metrics['r2']),    'R2_std':   np.std(metrics['r2']),
        'MAPE_mean': np.nanmean(metrics['mape']), 'MAPE_std': np.nanstd(metrics['mape']),
        'MaxErr_mean': np.mean(metrics['max_err']),
        'y_true': y_true_all, 'y_pred': y_pred_all,
        'best_params_per_fold': best_params_per_fold,
    }

print("✅ All training functions defined.")
""")

# =========================================================================
# CELL 9: Baseline Evaluation
# =========================================================================
md("## 8. Baseline Models (Linear Regression + Random Forest)")

code("""\
all_results = []

for subset_name, subset_df in subsets.items():
    X = subset_df[feature_cols]
    y = subset_df['Compressive_Strength']

    print(f"\\n── Baselines: {subset_name} ({len(X)} samples) ──")

    for name, model in [('LinearRegression', LinearRegression()),
                        ('RandomForest', RandomForestRegressor(n_estimators=200, max_depth=10,
                                                                random_state=RANDOM_SEED, n_jobs=-1))]:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        rmses, maes, r2s = [], [], []
        for tr_idx, te_idx in cv.split(X):
            m = type(model)(**model.get_params())
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            yp = m.predict(X.iloc[te_idx])
            rmses.append(np.sqrt(mean_squared_error(y.iloc[te_idx], yp)))
            maes.append(mean_absolute_error(y.iloc[te_idx], yp))
            r2s.append(r2_score(y.iloc[te_idx], yp))

        res = {'Subset': subset_name, 'Model': name,
               'RMSE_mean': np.mean(rmses), 'RMSE_std': np.std(rmses),
               'MAE_mean': np.mean(maes),   'MAE_std': np.std(maes),
               'R2_mean': np.mean(r2s),     'R2_std': np.std(r2s)}
        all_results.append(res)
        print(f"  {name:<20s} RMSE={res['RMSE_mean']:.3f}±{res['RMSE_std']:.3f}  R²={res['R2_mean']:.4f}±{res['R2_std']:.4f}")

print("\\n✅ Baselines complete.")
""")

# =========================================================================
# CELL 10: Main Training Loop
# =========================================================================
md("""\
## 9. Gradient Boosting Model Training

Runs **nested cross-validation** for XGBoost, CatBoost, and LightGBM on each age subset.
Each outer fold runs a fresh Optuna optimization (inner loop) — this gives **unbiased** performance estimates.

> ⏱ Expected time: ~15 min (quick/10 trials) · ~2-4 hours (full/100 trials) on T4 GPU
""")

code("""\
MODEL_TYPES = ['XGBoost', 'CatBoost', 'LightGBM']
all_best_params = {}
all_predictions = {}  # Store for plotting

total_combos = len(subsets) * len(MODEL_TYPES)
combo_idx = 0
pipeline_start = time.time()

for subset_name, subset_df in subsets.items():
    X = subset_df[feature_cols]
    y = subset_df['Compressive_Strength']

    for model_type in MODEL_TYPES:
        combo_idx += 1
        print(f"\\n{'='*60}")
        print(f"  [{combo_idx}/{total_combos}] {model_type} on {subset_name} ({len(X)} samples)")
        print(f"{'='*60}")
        t0 = time.time()

        results = nested_cv(X, y, model_type)

        elapsed = time.time() - t0
        print(f"\\n  Results ({elapsed/60:.1f} min):")
        print(f"    RMSE: {results['RMSE_mean']:.3f} ± {results['RMSE_std']:.3f} MPa")
        print(f"    MAE:  {results['MAE_mean']:.3f} ± {results['MAE_std']:.3f} MPa")
        print(f"    R²:   {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")
        print(f"    MAPE: {results['MAPE_mean']:.1f} ± {results['MAPE_std']:.1f}%")

        all_results.append({
            'Subset': subset_name, 'Model': model_type,
            'RMSE_mean': results['RMSE_mean'], 'RMSE_std': results['RMSE_std'],
            'MAE_mean': results['MAE_mean'],   'MAE_std': results['MAE_std'],
            'R2_mean': results['R2_mean'],     'R2_std': results['R2_std'],
        })

        best_p = results['best_params_per_fold'][0]
        all_best_params[f"{model_type}_{subset_name}"] = best_p
        all_predictions[f"{model_type}_{subset_name}"] = {
            'y_true': results['y_true'], 'y_pred': results['y_pred']
        }

total_elapsed = time.time() - pipeline_start
print(f"\\n{'='*60}")
print(f"  ✅ All training complete! Total: {total_elapsed/60:.1f} min")
print(f"{'='*60}")
""")

# =========================================================================
# CELL 11: Results Summary
# =========================================================================
md("## 10. Results Summary")

code("""\
results_df = pd.DataFrame(all_results).round(4)

# Separate table for boosting models
boost_df = results_df[results_df['Model'].isin(MODEL_TYPES)].copy()

print("\\n" + "="*80)
print("  GRADIENT BOOSTING MODELS — NESTED CV RESULTS")
print("="*80)
print(boost_df.to_string(index=False))

# Best model per subset
print(f"\\n{'─'*50}")
print("  Best model per subset (by R²):")
print(f"{'─'*50}")
for subset in boost_df['Subset'].unique():
    sdf = boost_df[boost_df['Subset'] == subset]
    best = sdf.loc[sdf['R2_mean'].idxmax()]
    print(f"  {subset:<6}: {best['Model']:<12} R²={best['R2_mean']:.4f}  RMSE={best['RMSE_mean']:.3f} MPa")

print(f"\\n\\n{'='*80}")
print("  ALL MODELS (including baselines)")
print("="*80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('outputs/stage_a_results_summary.csv', index=False)
with open('outputs/best_hyperparameters.json', 'w') as f:
    json.dump(all_best_params, f, indent=2)
print("\\n✅ Results saved to outputs/")
""")

# =========================================================================
# CELL 12: Comparison Plots
# =========================================================================
md("## 11. Comparison Visualizations")

code("""\
MODEL_COLORS = {
    'XGBoost': '#2196F3', 'CatBoost': '#FF5722', 'LightGBM': '#4CAF50',
    'LinearRegression': '#9E9E9E', 'RandomForest': '#FF9800',
}

# Prediction scatter plots for boosting models
fig_rows = len(subsets)
fig, axes = plt.subplots(fig_rows, 3, figsize=(18, 5.5 * fig_rows))
if fig_rows == 1:
    axes = axes.reshape(1, -1)

for i, subset_name in enumerate(subsets.keys()):
    for j, model_type in enumerate(MODEL_TYPES):
        key = f"{model_type}_{subset_name}"
        if key not in all_predictions:
            continue
        yt = np.array(all_predictions[key]['y_true'])
        yp = np.array(all_predictions[key]['y_pred'])

        ax = axes[i, j]
        color = MODEL_COLORS[model_type]
        ax.scatter(yt, yp, alpha=0.5, s=30, c=color, edgecolors='white', linewidths=0.3)
        lims = [min(yt.min(), yp.min())-2, max(yt.max(), yp.max())+2]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        ax.set_title(f"{model_type} — {subset_name}\\nR²={r2:.4f}, RMSE={rmse:.2f}", fontweight='bold', fontsize=11)
        ax.set_xlabel('Measured (MPa)')
        ax.set_ylabel('Predicted (MPa)')
        ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/prediction_scatter_all.png', dpi=200, bbox_inches='tight')
plt.show()

# R² heatmap
if len(subsets) > 1:
    pivot = boost_df.pivot_table(index='Model', columns='Subset', values='R2_mean')
    ordered = [c for c in ['EA1','EA7','EA14','Full'] if c in pivot.columns]
    pivot = pivot[ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.7, vmax=1.0,
                ax=ax, linewidths=0.5, annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    ax.set_title('R² Score — Models × Subsets', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/r2_heatmap.png', dpi=200, bbox_inches='tight')
    plt.show()

print("✅ Plots saved to outputs/")
""")

# =========================================================================
# CELL 13: SHAP Analysis
# =========================================================================
md("""\
## 12. SHAP Explainability Analysis

Computes SHAP values for each boosting model on the **Full** dataset.
Generates:
- Feature importance bar chart
- Beeswarm summary plot
- Dependence plots for top-3 features
""")

code("""\
# Train final models on Full dataset for SHAP
if 'Full' in subsets:
    X_full = subsets['Full'][feature_cols]
    y_full = subsets['Full']['Compressive_Strength']

    for model_type in MODEL_TYPES:
        key = f"{model_type}_Full"
        if key not in all_best_params:
            print(f"⚠️  Skipping SHAP for {model_type} (no Full params)")
            continue

        print(f"\\n── SHAP: {model_type} ──")
        best_p = all_best_params[key]
        model = create_model(model_type, best_p)
        model.fit(X_full, y_full)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_full)

        # Feature importance bar
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_full, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {model_type}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"outputs/shap_importance_{model_type}.png", dpi=200, bbox_inches='tight')
        plt.show()

        # Beeswarm
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_full, show=False)
        plt.title(f"SHAP Summary — {model_type}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"outputs/shap_summary_{model_type}.png", dpi=200, bbox_inches='tight')
        plt.show()

        # Top-3 dependence plots
        importance = np.abs(shap_vals).mean(axis=0)
        top3 = list(X_full.columns[np.argsort(importance)[-3:][::-1]])
        print(f"  Top-3 features: {top3}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax_i, feat in enumerate(top3):
            plt.sca(axes[ax_i])
            shap.dependence_plot(feat, shap_vals, X_full, ax=axes[ax_i], show=False)
            axes[ax_i].set_title(f"{feat}", fontweight='bold')
        plt.suptitle(f"SHAP Dependence — {model_type}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"outputs/shap_dependence_{model_type}.png", dpi=200, bbox_inches='tight')
        plt.show()
else:
    print("⚠️  Full subset not included — skipping SHAP analysis.")
    print("    Add 'Full' to SUBSET_NAMES and re-run to generate SHAP plots.")

print("\\n✅ SHAP analysis complete.")
""")

# =========================================================================
# CELL 14: Export Models
# =========================================================================
md("""\
## 13. Export Trained Models for Deployment

Saves all model artifacts needed for inference:
- Native model files (`.json`, `.cbm`, `.txt`)
- Universal pickle files (`.pkl`)
- Feature configuration (`feature_config.json`)
- Results summary and hyperparameters
""")

code("""\
print("Saving model artifacts for deployment...\\n")

# Train final models on each subset with best params and save
saved_models = {}

for subset_name, subset_df in subsets.items():
    X = subset_df[feature_cols]
    y = subset_df['Compressive_Strength']

    for model_type in MODEL_TYPES:
        key = f"{model_type}_{subset_name}"
        if key not in all_best_params:
            continue

        best_p = all_best_params[key]
        model = create_model(model_type, best_p)
        model.fit(X, y)

        prefix = f"models/{model_type}_{subset_name}"

        # Native format
        if model_type == 'XGBoost':
            model.save_model(f"{prefix}.json")
            fmt = "json"
        elif model_type == 'CatBoost':
            model.save_model(f"{prefix}.cbm")
            fmt = "cbm"
        elif model_type == 'LightGBM':
            model.booster_.save_model(f"{prefix}.txt")
            fmt = "txt"

        # Universal pickle
        joblib.dump(model, f"{prefix}.pkl")

        saved_models[key] = {'native': f"{prefix}.{fmt}", 'pkl': f"{prefix}.pkl"}
        print(f"  ✅ {key}: saved (.{fmt} + .pkl)")

# Feature configuration
feature_config = {
    'raw_features': ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
                     'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age'],
    'all_features': feature_cols,
    'target': 'Compressive_Strength',
    'n_engineered': len(feature_cols) - 8,
    'subsets_trained': list(subsets.keys()),
    'models_trained': MODEL_TYPES,
    'saved_models': saved_models,
    'best_hyperparameters': all_best_params,
}

with open('models/feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"\\n  ✅ Feature config saved to models/feature_config.json")

# Copy results
shutil.copy('outputs/stage_a_results_summary.csv', 'models/stage_a_results_summary.csv')
shutil.copy('outputs/best_hyperparameters.json', 'models/best_hyperparameters.json')

print(f"\\n📦 All artifacts saved to models/")
print(f"   Total files: {len(os.listdir('models'))}")
for f in sorted(os.listdir('models')):
    size = os.path.getsize(f'models/{f}')
    print(f"   {f:<45s} {size/1024:.1f} KB")
""")

# =========================================================================
# CELL 15: Download
# =========================================================================
md("""\
## 14. Download Model Artifacts

Downloads a ZIP file containing all trained models, configs, and results.
Use these files in your local Streamlit app for inference.
""")

code("""\
# Create ZIP of all artifacts
shutil.make_archive('Stage_A_Models', 'zip', '.', 'models')

# Also create a ZIP of outputs (plots)
shutil.make_archive('Stage_A_Outputs', 'zip', '.', 'outputs')

print("📥 Downloading model artifacts...")
from google.colab import files
files.download('Stage_A_Models.zip')

print("\\n📥 Downloading output plots...")
files.download('Stage_A_Outputs.zip')

print("\\n✅ Done! Unzip Stage_A_Models.zip locally for inference.")
print("   The models/ directory contains everything needed for the Streamlit app.")
""")

# =========================================================================
# Final: Assemble notebook JSON
# =========================================================================

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
        },
        "language_info": {
            "name": "python",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}

output_path = "Stage_A_Training.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook generated: {output_path}")
print(f"   {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code + "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
