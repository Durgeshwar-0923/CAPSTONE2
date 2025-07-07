import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Artifact directories
PLOT_DIR = "outputs/plots"
STAGE_DIR = "outputs/stages"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(STAGE_DIR, exist_ok=True)

def save_stage(df: pd.DataFrame, stage: str):
    path = os.path.join(STAGE_DIR, f"{stage}.csv")
    df.to_csv(path, index=False)
    logger.info(f"💾 Saved stage '{stage}' to {path}")

def save_corr_plot(corr_matrix: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f", square=True)
    plt.title("Correlation Matrix (After Variance Filtering)")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"📉 Correlation heatmap saved to {path}")

def select_features(
    df: pd.DataFrame,
    threshold: float = 0.01,
    drop_high_corr: bool = True,
    corr_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Select features using variance thresholding and optional correlation filtering.

    Args:
        df (pd.DataFrame): Input DataFrame (can include non-numeric).
        threshold (float): Variance threshold.
        drop_high_corr (bool): Whether to drop highly correlated features.
        corr_threshold (float): Correlation threshold to remove redundant features.

    Returns:
        pd.DataFrame: Reduced DataFrame with selected features.
    """
    df = df.copy()
    logger.info("📊 Starting feature selection...")

    # ─── Numeric columns only ───
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("⚠️ No numeric features found for selection.")
        return df

    # ─── Variance Threshold ───
    selector = VarianceThreshold(threshold=threshold)
    reduced_array = selector.fit_transform(numeric_df)
    retained_columns = numeric_df.columns[selector.get_support()]
    reduced_df = pd.DataFrame(reduced_array, columns=retained_columns, index=df.index)
    logger.info(f"🪙 {len(retained_columns)} features kept after variance filtering.")

    # ─── Correlation Filtering ───
    if drop_high_corr:
        corr_matrix = reduced_df.corr().abs()
        save_corr_plot(corr_matrix)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        reduced_df = reduced_df.drop(columns=to_drop)
        logger.info(f"🔗 Dropped {len(to_drop)} highly correlated features (r > {corr_threshold}).")

    # ─── Combine with non-numeric ───
    non_numeric_df = df.drop(columns=numeric_df.columns)
    final_df = pd.concat([reduced_df, non_numeric_df], axis=1)

    save_stage(final_df, "feature_selection")
    logger.info(f"✅ Feature selection complete. Final shape: {final_df.shape}")
    return final_df
