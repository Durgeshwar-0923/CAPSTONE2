import os
import joblib
import pandas as pd
from category_encoders import TargetEncoder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def target_encode(
    df: pd.DataFrame,
    target: str,
    cat_cols: list[str],
    return_encoder: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, TargetEncoder]:
    """
    Applies target encoding to specified categorical columns and saves encoder.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target variable.
        cat_cols (list): List of categorical columns to encode.
        return_encoder (bool): Whether to return the fitted encoder.

    Returns:
        pd.DataFrame or (DataFrame, TargetEncoder): Transformed data and encoder.
    """
    df = df.copy()

    if not cat_cols:
        logger.warning("⚠️ No categorical columns provided for target encoding.")
        return df

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    df[target] = df[target].fillna(df[target].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")
            logger.info(f"🛠️ Filled NaNs in '{col}' with 'Unknown' before encoding")

    encoder = TargetEncoder(cols=cat_cols)
    df[cat_cols] = encoder.fit_transform(df[cat_cols], df[target])

    # ─── Save encoder ──────────────────────────────────────────
    encoder_path = os.path.join(ARTIFACT_DIR, "target_encoder.pkl")
    joblib.dump(encoder, encoder_path)
    logger.info(f"💾 Saved target encoder → {encoder_path}")

    logger.info(f"🎯 Target-encoded: {cat_cols}")
    return (df, encoder) if return_encoder else df
