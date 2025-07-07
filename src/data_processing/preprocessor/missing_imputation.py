import os
import json
import joblib
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("❓ Imputing missing values...")
    df = df.copy()

    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    numerical_imputer = {}
    categorical_imputed_cols = []

    # ─── Categorical Imputation ───────────────────────────────
    for col in cat_cols:
        missing_count = df[col].isna().sum()
        if df[col].dtype.name != 'category':
            df[col] = df[col].astype('category')

        if "Unknown" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories("Unknown")

        if missing_count > 0:
            df[col] = df[col].fillna("Unknown")
            categorical_imputed_cols.append(col)
            logger.info(f"🟠 Categorical imputation: {col} → 'Unknown' ({missing_count} filled)")

    # ─── Numerical Imputation ────────────────────────────────
    for col in num_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            numerical_imputer[col] = median_val
            logger.info(f"🔵 Numerical imputation: {col} → median={median_val:.2f} ({missing_count} filled)")

    # ─── Save Artifacts ──────────────────────────────────────
    if numerical_imputer:
        joblib.dump(numerical_imputer, os.path.join(ARTIFACT_DIR, "numerical_imputer_map.pkl"))
        logger.info("💾 Saved numerical imputer map → artifacts/numerical_imputer_map.pkl")

    if categorical_imputed_cols:
        with open(os.path.join(ARTIFACT_DIR, "categorical_imputer_columns.json"), "w") as f:
            json.dump(categorical_imputed_cols, f)
        logger.info("💾 Saved categorical imputed columns → artifacts/categorical_imputer_columns.json")

    logger.info("✅ Missing value imputation complete.")
    return df
