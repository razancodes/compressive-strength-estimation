"""
Data loading and preprocessing for UCI Concrete Compressive Strength dataset.
"""

import pandas as pd
import numpy as np


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load concrete dataset and perform initial cleaning.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized column names.
    """
    df = pd.read_csv(filepath)

    # Standardize column names (handle various naming conventions)
    column_mapping = {
        'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast_Furnace_Slag',
        'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly_Ash',
        'Water  (component 4)(kg in a m^3 mixture)': 'Water',
        'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse_Aggregate',
        'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine_Aggregate',
        'Age (day)': 'Age',
        'Concrete compressive strength(MPa, megapascals) ': 'Compressive_Strength',
        'Concrete compressive strength(MPa, megapascals)': 'Compressive_Strength',
        # Simplified names (in case the CSV uses them)
        'Cement': 'Cement',
        'Blast Furnace Slag': 'Blast_Furnace_Slag',
        'Fly Ash': 'Fly_Ash',
        'Water': 'Water',
        'Superplasticizer': 'Superplasticizer',
        'Coarse Aggregate': 'Coarse_Aggregate',
        'Fine Aggregate': 'Fine_Aggregate',
        'Age': 'Age',
        'Compressive Strength': 'Compressive_Strength',
    }

    # Rename columns that exist in the mapping
    new_columns = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in column_mapping:
            new_columns[col] = column_mapping[col_stripped]
        elif col in column_mapping:
            new_columns[col] = column_mapping[col]

    if new_columns:
        df = df.rename(columns=new_columns)

    # If columns are still not standardized, try positional assignment
    expected_cols = [
        'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
        'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate',
        'Age', 'Compressive_Strength'
    ]

    if not all(c in df.columns for c in expected_cols):
        if len(df.columns) == 9:
            print("WARNING: Column names not recognized. Assigning by position.")
            df.columns = expected_cols
        else:
            raise ValueError(
                f"Expected 9 columns, got {len(df.columns)}. "
                f"Columns found: {list(df.columns)}"
            )

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values detected:")
        print(missing[missing > 0])
        print("Dropping rows with missing values...")
        df = df.dropna()

    # Remove exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"Removed {n_removed} duplicate rows.")

    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget statistics (Compressive Strength, MPa):")
    print(f"  Min:    {df['Compressive_Strength'].min():.2f}")
    print(f"  Max:    {df['Compressive_Strength'].max():.2f}")
    print(f"  Mean:   {df['Compressive_Strength'].mean():.2f}")
    print(f"  Median: {df['Compressive_Strength'].median():.2f}")
    print(f"  Std:    {df['Compressive_Strength'].std():.2f}")

    # Age distribution
    print(f"\nAge distribution (days):")
    print(f"  Unique ages: {sorted(df['Age'].unique())}")
    print(f"  Samples per age:")
    age_counts = df['Age'].value_counts().sort_index()
    for age, count in age_counts.items():
        print(f"    {int(age):>4d} days: {count} samples")

    return df


def create_age_subsets(df: pd.DataFrame) -> dict:
    """
    Create early-age prediction subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'Age' column.

    Returns
    -------
    dict
        Dictionary with subset names as keys and DataFrames as values.
    """
    subsets = {
        'EA1': df[df['Age'] <= 3].copy(),
        'EA7': df[df['Age'] <= 7].copy(),
        'EA14': df[df['Age'] <= 14].copy(),
        'Full': df.copy(),
    }

    print("\nEarly-Age Subset Sizes:")
    print("-" * 45)
    print(f"{'Subset':<10} {'Age Range':<15} {'Samples':>8}")
    print("-" * 45)
    for name, subset in subsets.items():
        age_range = f"1-{int(subset['Age'].max())}d"
        print(f"{name:<10} {age_range:<15} {len(subset):>8}")
    print("-" * 45)

    return subsets
