"""
Publication-quality visualization functions for Stage A results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Global style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ── Color palette ───────────────────────────────────────────────────────────

MODEL_COLORS = {
    'XGBoost': '#2196F3',       # Blue
    'CatBoost': '#FF5722',      # Deep orange
    'LightGBM': '#4CAF50',      # Green
    'LinearRegression': '#9E9E9E',  # Grey
    'RandomForest': '#FF9800',  # Orange
}


# ── Prediction scatter plot ────────────────────────────────────────────────

def plot_predictions(
    y_true, y_pred, model_name: str, subset_name: str,
    output_dir: str = 'outputs',
):
    """Predicted vs Measured scatter plot with metric annotations."""
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    color = MODEL_COLORS.get(model_name, '#607D8B')

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.55, edgecolors='white',
               linewidths=0.5, s=55, c=color, zorder=3)

    # Perfect prediction line
    lims = [
        min(y_true.min(), y_pred.min()) - 2,
        max(y_true.max(), y_pred.max()) + 2,
    ]
    ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.7, label='Perfect prediction')

    # ±10% error band
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.fill_between(x_line, x_line * 0.9, x_line * 1.1,
                     alpha=0.08, color='grey', label='±10% error band')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Measured Compressive Strength (MPa)')
    ax.set_ylabel('Predicted Compressive Strength (MPa)')
    ax.set_title(f'{model_name} — {subset_name}', fontweight='bold')

    # Metrics annotation
    textstr = (f'R² = {r2:.4f}\n'
               f'RMSE = {rmse:.2f} MPa\n'
               f'MAE = {mae:.2f} MPa\n'
               f'n = {len(y_true)}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 alpha=0.85, edgecolor='grey')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    ax.grid(alpha=0.2)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"pred_scatter_{model_name}_{subset_name}.png"))
    plt.close(fig)


# ── Residual plot ───────────────────────────────────────────────────────────

def plot_residuals(
    y_true, y_pred, model_name: str, subset_name: str,
    output_dir: str = 'outputs',
):
    """Residual scatter + histogram."""
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_pred - y_true

    color = MODEL_COLORS.get(model_name, '#607D8B')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Residual scatter
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.55, edgecolors='white',
               linewidths=0.5, s=50, c=color, zorder=3)
    ax.axhline(0, color='k', linestyle='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('Predicted Strength (MPa)')
    ax.set_ylabel('Residuals (MPa)')
    ax.set_title(f'Residual Plot — {model_name} ({subset_name})',
                 fontweight='bold')
    ax.grid(alpha=0.2)

    # Right: Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, edgecolor='white', alpha=0.8, color=color)
    ax.axvline(0, color='k', linestyle='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('Residuals (MPa)')
    ax.set_ylabel('Frequency')
    ax.set_title(
        f'Residual Distribution\n'
        f'Mean = {np.mean(residuals):.2f} MPa, '
        f'Std = {np.std(residuals):.2f} MPa',
        fontweight='bold',
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"residuals_{model_name}_{subset_name}.png"))
    plt.close(fig)


# ── Model comparison bar chart ──────────────────────────────────────────────

def plot_model_comparison(
    results_df: pd.DataFrame,
    output_dir: str = 'outputs',
):
    """
    Grouped bar chart comparing R², RMSE, MAE across models and subsets.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ('R2_mean', 'R2_std', 'R² Score', True),
        ('RMSE_mean', 'RMSE_std', 'RMSE (MPa)', False),
        ('MAE_mean', 'MAE_std', 'MAE (MPa)', False),
    ]

    subsets_order = ['EA1', 'EA7', 'EA14', 'Full']
    models_order = results_df['Model'].unique()

    for ax, (mean_col, std_col, ylabel, higher_better) in zip(axes, metrics):
        x = np.arange(len(subsets_order))
        width = 0.2
        n_models = len(models_order)

        for i, model in enumerate(models_order):
            model_data = results_df[results_df['Model'] == model]
            means, stds = [], []
            for subset in subsets_order:
                row = model_data[model_data['Subset'] == subset]
                if len(row) > 0:
                    means.append(row[mean_col].values[0])
                    stds.append(row[std_col].values[0])
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - n_models / 2 + 0.5) * width
            color = MODEL_COLORS.get(model, '#607D8B')
            ax.bar(x + offset, means, width, yerr=stds,
                   label=model, color=color, alpha=0.85,
                   edgecolor='white', capsize=3)

        ax.set_xlabel('Subset')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subsets_order)
        ax.legend()
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Model Performance Comparison Across Age Subsets',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close(fig)


# ── Subset performance heatmap ──────────────────────────────────────────────

def plot_performance_heatmap(
    results_df: pd.DataFrame,
    output_dir: str = 'outputs',
):
    """Heatmap of R² scores across models and subsets."""
    os.makedirs(output_dir, exist_ok=True)

    subsets_order = ['EA1', 'EA7', 'EA14', 'Full']
    pivot = results_df.pivot_table(
        index='Model', columns='Subset', values='R2_mean',
    )
    # Reorder columns
    pivot = pivot[[c for c in subsets_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0.7, vmax=1.0, ax=ax,
        linewidths=0.5, linecolor='white',
        annot_kws={'fontsize': 13, 'fontweight': 'bold'},
    )
    ax.set_title('R² Score — Models × Subsets', fontweight='bold', fontsize=14)
    ax.set_xlabel('Age Subset')
    ax.set_ylabel('Model')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "r2_heatmap.png"))
    plt.close(fig)


# ── EDA plots ──────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame, output_dir: str = 'outputs'):
    """Generate exploratory data analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['Compressive_Strength'], bins=40, edgecolor='white',
            alpha=0.8, color='#2196F3')
    ax.set_xlabel('Compressive Strength (MPa)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Compressive Strength', fontweight='bold')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_target_dist.png"))
    plt.close(fig)

    # 2. Correlation heatmap
    raw_cols = [
        'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
        'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate',
        'Age', 'Compressive_Strength',
    ]
    cols_present = [c for c in raw_cols if c in df.columns]
    corr = df[cols_present].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, ax=ax, linewidths=0.5,
        annot_kws={'fontsize': 9},
    )
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_correlation.png"))
    plt.close(fig)

    # 3. Strength vs Age
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['Age'], df['Compressive_Strength'], alpha=0.4,
               s=30, c='#FF5722', edgecolors='white', linewidths=0.3)
    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Compressive Strength (MPa)')
    ax.set_title('Compressive Strength vs Curing Age', fontweight='bold')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_strength_vs_age.png"))
    plt.close(fig)

    print(f"  EDA plots saved to {output_dir}/")
