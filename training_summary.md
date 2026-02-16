# Stage A: Training Summary
## Early-Age Concrete Compressive Strength Prediction — ML Baseline

**Date:** February 16, 2026  
**Runtime:** 12.7 hours (760 min) on CPU  
**Dataset:** UCI Concrete — 1,005 samples (25 duplicates removed)  
**Features:** 22 (8 raw + 14 domain-engineered)  
**Validation:** 5-fold Nested CV with 100 Optuna TPE trials per inner loop  

---

## Dataset Statistics

| Parameter | Value |
|---|---|
| Samples | 1,005 |
| Target range | 2.33 – 82.60 MPa |
| Target mean ± std | 35.25 ± 16.28 MPa |
| Age values | 1, 3, 7, 14, 28, 56, 90, 91, 100, 120, 180, 270, 360, 365 days |
| Most common age | 28 days (419 samples, 42%) |

### Age Subsets

| Subset | Age Range | Samples | Purpose |
|---|---|---|---|
| EA1 | ≤ 3 days | 131 | Formwork removal decisions |
| EA7 | ≤ 7 days | 253 | Construction sequencing |
| EA14 | ≤ 14 days | 315 | Quality assurance |
| Full | 1–365 days | 1,005 | Baseline comparison |

---

## Baseline Models (5-Fold CV)

| Subset | Linear Regression |  | Random Forest |  |
|---|---|---|---|---|
| | RMSE (MPa) | R² | RMSE (MPa) | R² |
| EA1 | 3.79 ± 0.54 | 0.827 | 4.64 ± 0.57 | 0.747 |
| EA7 | 4.86 ± 0.64 | 0.838 | 5.70 ± 1.30 | 0.769 |
| EA14 | 5.04 ± 0.35 | 0.823 | 5.07 ± 0.78 | 0.819 |
| Full | 6.86 ± 0.37 | 0.821 | 4.75 ± 0.26 | 0.914 |

---

## Gradient Boosting Models — Nested CV Results

### Full Dataset (1,005 samples)

| Model | RMSE (MPa) | MAE (MPa) | R² | MAPE | Time |
|---|---|---|---|---|---|
| XGBoost | 4.06 ± 0.24 | 2.80 ± 0.15 | 0.937 ± 0.007 | 9.8% | 13 min |
| **CatBoost** | **3.72 ± 0.29** | **2.48 ± 0.09** | **0.947 ± 0.007** | **8.6%** | **311 min** |
| LightGBM | 4.08 ± 0.49 | 2.82 ± 0.26 | 0.937 ± 0.011 | 9.8% | 22 min |

### EA14 — Early Age ≤ 14 days (315 samples)

| Model | RMSE (MPa) | MAE (MPa) | R² | MAPE | Time |
|---|---|---|---|---|---|
| XGBoost | 4.80 ± 0.60 | 3.34 ± 0.31 | 0.837 ± 0.042 | 17.4% | 6 min |
| **CatBoost** | **4.27 ± 0.71** | **2.94 ± 0.27** | **0.870 ± 0.044** | **14.8%** | **138 min** |
| LightGBM | 5.09 ± 0.59 | 3.58 ± 0.33 | 0.815 ± 0.050 | 18.7% | 9 min |

### EA7 — Early Age ≤ 7 days (253 samples)

| Model | RMSE (MPa) | MAE (MPa) | R² | MAPE | Time |
|---|---|---|---|---|---|
| XGBoost | 5.29 ± 1.30 | 3.63 ± 0.78 | 0.798 ± 0.089 | 19.2% | 6 min |
| **CatBoost** | **4.71 ± 0.92** | **3.14 ± 0.57** | **0.840 ± 0.070** | **16.7%** | **134 min** |
| LightGBM | 5.38 ± 1.00 | 3.76 ± 0.64 | 0.794 ± 0.073 | 20.9% | 9 min |

### EA1 — Early Age ≤ 3 days (131 samples)

| Model | RMSE (MPa) | MAE (MPa) | R² | MAPE | Time |
|---|---|---|---|---|---|
| XGBoost | 4.48 ± 0.72 | 3.29 ± 0.39 | 0.758 ± 0.088 | 21.2% | 5 min |
| **CatBoost** | **4.16 ± 0.43** | **3.03 ± 0.35** | **0.794 ± 0.060** | **19.3%** | **102 min** |
| LightGBM | 4.60 ± 0.31 | 3.46 ± 0.32 | 0.751 ± 0.059 | 23.8% | 3 min |

---

## Best Model Per Subset

| Subset | Champion | R² | RMSE (MPa) |
|---|---|---|---|
| EA1 (≤ 3 days) | Linear Regression* | 0.827 | 3.79 |
| EA7 (≤ 7 days) | CatBoost | 0.840 | 4.71 |
| EA14 (≤ 14 days) | CatBoost | 0.870 | 4.27 |
| Full (all ages) | CatBoost | 0.947 | 3.72 |

> *Linear Regression wins on EA1 due to limited sample size (131) and minimal age variation (only 1 and 3 days), where simpler models avoid overfitting.

---

## Group-Based CV (Mix-Level Generalization)

Tests generalization to **entirely unseen mix designs** using GroupKFold:

| Model | RMSE (MPa) | R² |
|---|---|---|
| XGBoost | 5.311 | 0.892 |
| **CatBoost** | **5.141** | **0.898** |
| LightGBM | 5.361 | 0.889 |

> The ~5% R² drop from random CV → group CV is expected and confirms the model generalizes to new mixes, not just new specimens of seen mixes.

---

## SHAP Feature Importance (Top-3 per Model × Subset)

| Subset | XGBoost | CatBoost | LightGBM |
|---|---|---|---|
| EA1 | W/B ratio, Cement, Binder | Binder, Cement, W/B ratio | W/B ratio, Cement, Binder |
| EA7 | W/B ratio, Cement, Age | Cement, W/B ratio, Binder | W/B ratio, Age, Cement |
| EA14 | W/B ratio, Age, Cement | W/B ratio, Cement, Binder | W/B ratio, Age, Cement |
| Full | Age, W/B ratio, Binder | W/B ratio, Age, log(Age) | Age, W/B ratio, Binder |

**Key insight:** W/B ratio is the most consistent top predictor across all models and subsets, aligning with fundamental concrete engineering theory (Abrams' law). Age becomes dominant on the Full dataset where it spans 1–365 days.

---

## Literature Comparison

| Metric | Our Result (CatBoost, Full) | Best Literature | Gap |
|---|---|---|---|
| R² | 0.947 | 0.953 (CatXG Hybrid, 2024) | −0.006 |
| RMSE | 3.72 MPa | 3.06 MPa (CatBoost, Nature 2024) | +0.66 |
| MAE | 2.48 MPa | 2.26 MPa (CatXG, 2024) | +0.22 |

> The gap is expected: our results use **nested CV** (unbiased generalization estimate), while most papers use standard train/test splits or non-nested CV, which produce optimistically biased metrics.

---

## Outputs Generated

| Category | Count | Location |
|---|---|---|
| Prediction scatter plots | 20 | `outputs/pred_scatter_*.png` |
| Residual plots | 20 | `outputs/residuals_*.png` |
| SHAP importance plots | 12 | `outputs/shap_importance_*.png` |
| SHAP summary plots | 12 | `outputs/shap_summary_*.png` |
| SHAP dependence plots | 36 | `outputs/shap_dep_*.png` |
| EDA plots | 3 | `outputs/eda_*.png` |
| Model comparison | 1 | `outputs/model_comparison.png` |
| R² heatmap | 1 | `outputs/r2_heatmap.png` |
| Results CSV | 1 | `outputs/stage_a_results_summary.csv` |
| Hyperparameters | 1 | `outputs/best_hyperparameters.json` |
| **Total** | **107** | |
