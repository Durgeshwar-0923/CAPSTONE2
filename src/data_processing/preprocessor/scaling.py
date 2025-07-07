import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def scale_features(df: pd.DataFrame, exclude_cols: list[str] = None, return_scaler: bool = False):
    """
    Scales all numeric features using StandardScaler, excluding specified columns.
    Also saves both feature and target scalers if 'adjusted_total_usd' is excluded.

    Args:
        df (pd.DataFrame): Input DataFrame.
        exclude_cols (list, optional): List of numeric columns to exclude from feature scaling.
        return_scaler (bool, optional): If True, returns a tuple (scaled_df, feature_scaler).

    Returns:
        pd.DataFrame or tuple: Scaled DataFrame, optionally with fitted feature scaler.
    """
    logger.info("📏 Scaling numerical features...")
    df = df.copy()
    exclude_cols = exclude_cols or []

    # Scale target if excluded
    if "adjusted_total_usd" in exclude_cols and "adjusted_total_usd" in df.columns:
        target_scaler = StandardScaler()
        df["adjusted_total_usd"] = target_scaler.fit_transform(df[["adjusted_total_usd"]])
        target_scaler_path = os.path.join(ARTIFACT_DIR, "target_scaler.pkl")
        joblib.dump(target_scaler, target_scaler_path)
        logger.info(f"📦 Saved target scaler to '{target_scaler_path}'")

    # Identify numeric columns to scale (excluding excluded columns)
    num_cols = df.select_dtypes(include="number").columns.difference(exclude_cols).tolist()
    if not num_cols:
        logger.warning("⚠️ No numeric features to scale.")
        if return_scaler:
            return df, None
        else:
            return df

    feature_scaler = StandardScaler()
    df[num_cols] = feature_scaler.fit_transform(df[num_cols])
    feature_scaler_path = os.path.join(ARTIFACT_DIR, "feature_scaler.pkl")
    joblib.dump(feature_scaler, feature_scaler_path)
    logger.info(f"📐 Scaled features: {num_cols}")
    logger.info(f"📦 Saved feature scaler to '{feature_scaler_path}'")
    logger.info("✅ Scaling complete.")

    if return_scaler:
        return df, feature_scaler
    else:
        return df
