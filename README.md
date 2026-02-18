# Early-Age Concrete Compressive Strength Prediction

Machine learning models for predicting the compressive strength of concrete at early ages, trained mostly on the UCI Concrete dataset using gradient boosting with rigorous nested cross-validation.

**Live App:** [compressive-strength-estimation.streamlit.app](https://compressive-strength-estimation.streamlit.app/)

---

## Overview

Accurate early-age strength prediction enables informed decisions about formwork removal timing, construction sequencing, and quality assurance — reducing reliance on destructive testing and accelerating construction schedules.

This project trains and evaluates 12 gradient boosting models (XGBoost, CatBoost, LightGBM) across four age-based subsets:

| Subset | Age Range | Samples | Application |
|--------|-----------|---------|-------------|
| EA1 | Up to 3 days | 131 | Formwork removal decisions |
| EA7 | Up to 7 days | 253 | Construction sequencing |
| EA14 | Up to 14 days | 315 | Quality assurance checks |
| Full | 1 to 365 days | 1,005 | General strength estimation |

### What the Model Predicts

Given a concrete mix design (cement, water, aggregates, supplementary cementitious materials, superplasticizer) and a curing age, the model predicts the 28-day (or any age) compressive strength in MPa. The best model (CatBoost, Full subset) achieves:

- **R-squared = 0.947** (explains 94.7% of strength variance)
- **RMSE = 3.72 MPa** (average error under 4 MPa)
- **MAE = 2.48 MPa** (typical prediction within 2.5 MPa of actual)

The model also provides SHAP-based explanations showing which mix design factors most influenced each prediction.

---

## Results

### Best Model Performance (Nested 5-Fold CV)

| Subset | Best Model | RMSE (MPa) | MAE (MPa) | R-squared | MAPE |
|--------|------------|------------|-----------|-----------|------|
| EA1 | CatBoost | 4.16 +/- 0.43 | 3.03 +/- 0.35 | 0.794 | 19.3% |
| EA7 | CatBoost | 4.71 +/- 0.92 | 3.14 +/- 0.57 | 0.840 | 16.7% |
| EA14 | CatBoost | 4.27 +/- 0.71 | 2.94 +/- 0.27 | 0.870 | 14.8% |
| Full | CatBoost | 3.72 +/- 0.29 | 2.48 +/- 0.09 | 0.947 | 8.6% |

CatBoost consistently outperforms XGBoost and LightGBM across all subsets.

### Mix-Level Generalization (Group-Based CV)

Testing whether the model generalizes to entirely unseen mix designs, not just new specimens of previously seen mixes:

| Model | RMSE (MPa) | R-squared |
|-------|------------|-----------|
| XGBoost | 5.311 | 0.892 |
| CatBoost | 5.141 | 0.898 |
| LightGBM | 5.361 | 0.889 |

The approximately 5% R-squared drop from random CV to group CV confirms the model generalizes to novel mix designs.

### Literature Comparison

| Metric | This Study (CatBoost) | Best Published | Note |
|--------|----------------------|----------------|------|
| R-squared | 0.947 | 0.953 | CatXG Hybrid, 2024 |
| RMSE | 3.72 MPa | 3.06 MPa | CatBoost, Nature 2024 |
| MAE | 2.48 MPa | 2.26 MPa | CatXG, 2024 |

Our results use nested CV (unbiased generalization estimate). Most published results use standard train/test splits, which produce optimistically biased metrics.

---

## Key Findings

### Feature Importance (SHAP Analysis)

The water-to-binder ratio (W/B) is the most consistent top predictor across all models and subsets, aligning with Abrams' law from fundamental concrete engineering:

| Subset | Top-3 Drivers (CatBoost) |
|--------|--------------------------|
| EA1 | Binder, Cement, W/B ratio |
| EA7 | Cement, W/B ratio, Binder |
| EA14 | W/B ratio, Cement, Binder |
| Full | W/B ratio, Age, log(Age) |

Age becomes the dominant predictor on the Full dataset where it spans 1 to 365 days.

### Engineered Features

14 domain-engineered features are derived from the 8 raw inputs, bringing the total to 22 model features:

- **Binder system:** Binder content, W/B ratio, GGBS ratio, Fly Ash ratio, SCM ratio
- **Aggregate system:** Total aggregate, Fine aggregate ratio, Aggregate-to-binder ratio
- **Admixture:** Superplasticizer per binder
- **Temporal:** log(Age), sqrt(Age), Age category indicators (very early, early, standard)

---

## Interactive Application

The Streamlit application provides five tools:

| Page | Description |
|------|-------------|
| **Predict** | Enter a mix design, get an instant strength prediction with SHAP explanation |
| **Strength Curve** | Project strength development from 1 to 365 days for a single mix |
| **Compare Models** | Run all 12 model variants on the same input and compare results |
| **Training Results** | Browse all 107 training plots: metrics, scatter, residuals, SHAP |
| **Batch Predict** | Upload a CSV of mix designs and download predictions |

Missing input values are automatically filled with dataset-verified defaults (training data means for mandatory fields, zero for optional SCMs and admixtures).

**Try it:** [compressive-strength-estimation.streamlit.app](https://compressive-strength-estimation.streamlit.app/)

---

## Dataset

**Source:** UCI Concrete Compressive Strength Dataset (Yeh, 1998)

| Parameter | Value |
|-----------|-------|
| Samples | 1,005 (25 duplicates removed from original 1,030) |
| Target range | 2.33 to 82.60 MPa |
| Target mean | 35.25 +/- 16.28 MPa |
| Age values | 1, 3, 7, 14, 28, 56, 90, 91, 100, 120, 180, 270, 360, 365 days |
| Most common age | 28 days (419 samples, 42%) |

### Input Variables

| Variable | Unit | Range |
|----------|------|-------|
| Cement | kg/m3 | 102 to 540 |
| Blast Furnace Slag | kg/m3 | 0 to 359 |
| Fly Ash | kg/m3 | 0 to 200 |
| Water | kg/m3 | 122 to 247 |
| Superplasticizer | kg/m3 | 0 to 32 |
| Coarse Aggregate | kg/m3 | 801 to 1,145 |
| Fine Aggregate | kg/m3 | 594 to 993 |
| Age | days | 1 to 365 |

---

## Methodology

- **Models:** XGBoost, CatBoost, LightGBM
- **Hyperparameter Optimization:** Optuna TPE sampler, 100 trials per model-subset combination
- **Validation:** 5-fold nested cross-validation (unbiased performance estimation)
- **Baselines:** Linear Regression, Random Forest
- **Explainability:** SHAP TreeExplainer (importance, summary, and dependence plots)
- **Training Time:** 12.7 hours total on CPU (Google Colab)

---

## Project Structure

```
compressive-strength-estimation/
├── app/                           # Streamlit application
│   ├── app.py                     # Entry point
│   ├── model_loader.py            # Model loading, feature engineering, SHAP
│   ├── styles.py                  # Shared CSS
│   └── pages/
│       ├── 1_Predict.py           # Single prediction + SHAP
│       ├── 2_Strength_Curve.py    # Multi-age projection
│       ├── 3_Compare_Models.py    # 12-model comparison
│       ├── 4_Training_Results.py  # Training dashboard (107 plots)
│       └── 5_Batch_Predict.py     # CSV upload/download
├── data/                          # UCI Concrete dataset
├── models/                        # 12 trained model weights (.pkl)
├── outputs/                       # 107 plots, metrics CSV, hyperparameters
├── src/                           # Training pipeline modules
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── optimization.py
│   ├── validation.py
│   ├── explainability.py
│   └── visualization.py
├── main.py                        # Training pipeline orchestrator
├── requirements.txt               # Training dependencies
├── requirements_app.txt           # Streamlit app dependencies
└── README.md
```

---

## Local Setup

### Run the Streamlit App

```bash
pip install -r requirements_app.txt
streamlit run app/app.py
```

### Retrain Models

```bash
pip install -r requirements.txt

# Quick run (10 Optuna trials, ~10 min)
python main.py --quick

# Full run (100 Optuna trials, ~7 hours)
python main.py
```

---

## References

1. Yeh, I-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research*, 28(12), 1797-1808.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
3. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*.
4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
5. Akiba, T., et al. (2019). Optuna: A Next-Generation Hyperparameter Optimization Framework. *KDD*.
6. Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

## Author

**razancodes**
- GitHub: [github.com/razancodes](https://github.com/razancodes)
- Email: razancodes@gmail.com
