# src/data_processing/preprocessor/outlier_handler.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def handle_outliers(df: pd.DataFrame, method="iqr", multiplier=1.5, z_thresh=3.0) -> pd.DataFrame:
    logger.info(f"🚫 Handling outliers using {method.upper()} method...")
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        col_data = df[col]
        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
        elif method == "zscore":
            mean = col_data.mean()
            std = col_data.std()
            lower = mean - z_thresh * std
            upper = mean + z_thresh * std
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        outliers = ((col_data < lower) | (col_data > upper)).sum()
        df[col] = col_data.clip(lower, upper)
        logger.info(f"📉 {col}: clipped {outliers} values between [{lower:.2f}, {upper:.2f}]")

    logger.info("✅ Outlier handling complete.")
    return df

def plot_distribution_comparison(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    column: str,
    bins: int = 50,
    save_path: str = None
):
    """
    Plot the distribution of a numeric column before and after outlier handling.

    Args:
        original (pd.DataFrame): Data before outlier handling.
        cleaned (pd.DataFrame): Data after outlier handling.
        column (str): Column to plot.
        bins (int): Histogram bins.
        save_path (str): Optional path to save the plot.
    """
    if column not in original.columns or column not in cleaned.columns:
        logger.warning(f"❌ Column '{column}' not found in input dataframes.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i, (df_plot, title) in enumerate(zip([original, cleaned], ["Before", "After"])):
        sns.histplot(df_plot[column].dropna(), bins=bins, kde=True, ax=ax[i], color="skyblue")
        ax[i].axvline(df_plot[column].mean(), color='red', linestyle='--', label='Mean')
        ax[i].axvline(df_plot[column].median(), color='green', linestyle=':', label='Median')
        ax[i].set_title(f"{title} Outlier Handling - {column}")
        ax[i].set_xlabel(column)
        ax[i].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"📸 Distribution plot saved to {save_path}")
    plt.show()