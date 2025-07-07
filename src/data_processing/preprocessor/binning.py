import os
import joblib
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def bin_feature(df: pd.DataFrame, column: str, bins=5, labels=None, strategy='quantile') -> pd.DataFrame:
    """
    Bin a numeric column into discrete intervals and save bin config.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column to bin.
        bins (int or list): Number of bins or specific bin edges.
        labels (list, optional): Labels for the resulting bins.
        strategy (str): 'quantile' or 'uniform'.

    Returns:
        pd.DataFrame: DataFrame with new binned feature column.
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"❌ Column '{column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"❌ Column '{column}' must be numeric for binning.")

    new_col = column + "_bin"
    bin_edges = None

    try:
        if strategy == 'quantile':
            df[new_col], bin_edges = pd.qcut(df[column], q=bins, labels=labels, retbins=True, duplicates="drop")
            logger.info(f"📦 Quantile binning applied to '{column}' → '{new_col}'")
        elif strategy == 'uniform':
            df[new_col], bin_edges = pd.cut(df[column], bins=bins, labels=labels, retbins=True)
            logger.info(f"📦 Uniform binning applied to '{column}' → '{new_col}'")
        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'")
    except ValueError as e:
        logger.warning(f"⚠️ Binning failed for column '{column}': {e}")
        df[new_col] = pd.NA

    # Handle NaNs created during binning
    if df[new_col].isnull().any():
        df[new_col] = df[new_col].cat.add_categories(['Unknown'])
        df[new_col] = df[new_col].fillna('Unknown')
        logger.info(f"🛠️ Filled NaNs in '{new_col}' with 'Unknown'")

    # Save binning configuration
    config = {
        "column": column,
        "strategy": strategy,
        "bins": bin_edges,
        "labels": labels,
    }
    joblib.dump(config, os.path.join(ARTIFACT_DIR, f"binning_{column}.pkl"))
    logger.info(f"💾 Saved binning config for '{column}' → artifacts/binning_{column}.pkl")

    return df
