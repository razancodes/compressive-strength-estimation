# Stage A: Early-Age Compressive Strength Prediction – ML Baseline

Data-driven early-age concrete strength prediction using XGBoost, CatBoost, and LightGBM with Optuna hyperparameter optimization, nested cross-validation, and SHAP explainability.

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Data

Place `Concrete_Data.csv` in the `data/` directory. This is the UCI Concrete Compressive Strength dataset (Yeh, 1998) with 1030 samples and 9 columns.

## Usage

```bash
# Quick smoke test (~5-10 min, 10 Optuna trials)
python main.py --quick

# Full run (~4-7 hours, 100 Optuna trials)
python main.py

# Custom trial count
python main.py --n-trials 50
```

## Outputs

All results are saved in `outputs/`:
- `stage_a_results_summary.csv` – Performance metrics (RMSE, MAE, R²) for all model-subset combinations
- `best_hyperparameters.json` – Optimized hyperparameters per model-subset
- `shap_*.png` – SHAP feature importance, summary, and dependence plots
- `pred_scatter_*.png` – Predicted vs Measured scatter plots
- `residuals_*.png` – Residual analysis plots

## Project Structure

```
├── data/                      # Dataset (user-provided)
├── src/
│   ├── data_loader.py         # Data loading & subset creation
│   ├── feature_engineering.py # Domain-derived feature construction
│   ├── optimization.py        # Optuna hyperparameter objectives
│   ├── validation.py          # Nested CV & evaluation
│   ├── explainability.py      # SHAP analysis
│   └── visualization.py       # Publication-quality plots
├── outputs/                   # Generated results
├── main.py                    # Pipeline orchestrator
├── requirements.txt           # Dependencies
└── README.md
```

## Methodology

- **Models:** XGBoost, CatBoost, LightGBM
- **HPO:** Optuna (TPE sampler, 100 trials default)
- **Validation:** 5-fold nested cross-validation (unbiased performance estimation)
- **Subsets:** EA1 (≤3d), EA7 (≤7d), EA14 (≤14d), Full (all ages)
- **Explainability:** SHAP TreeExplainer with importance, summary, and dependence plots
- **Baselines:** Linear Regression, Random Forest

## References

- Yeh, I-C. (1998). UCI Concrete Compressive Strength Dataset.
- Chen & Guestrin (2016). XGBoost.
- Prokhorenkova et al. (2018). CatBoost.
- Ke et al. (2017). LightGBM.
- Akiba et al. (2019). Optuna.
- Lundberg & Lee (2017). SHAP.
