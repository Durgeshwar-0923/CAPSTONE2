import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import mlflow

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Output paths
OUTPUT_STAGE_DIR = "outputs/stages"
IMPORTANCE_PLOT_PATH = "outputs/importance/shap_feature_importance.png"
IMPORTANCE_CSV_PATH = "outputs/importance/shap_feature_importance.csv"

os.makedirs(OUTPUT_STAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)

def save_stage(df: pd.DataFrame, stage: str):
    """Saves intermediate feature engineering stages."""
    path = os.path.join(OUTPUT_STAGE_DIR, f"{stage}.csv")
    df.to_csv(path, index=False)
    logger.info(f"💾 Saved stage '{stage}' to {path}")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🧠 Applying feature engineering...")
    df = df.copy()

    if "adjusted_total_usd" in df.columns and "years_of_experience" in df.columns:
        df["salary_per_year"] = df["adjusted_total_usd"] / (df["years_of_experience"] + 1)
        df["log_adjusted_total_usd"] = np.log1p(df["adjusted_total_usd"])
        logger.info("➕ Created 'salary_per_year' and 'log_adjusted_total_usd'.")

    save_stage(df, "feature_engineering")
    return df

def apply_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("💡 Applying domain-specific features...")
    df = df.copy()

    base_salary = df.get("base_salary", pd.Series(np.ones(len(df))))
    base_salary_safe = base_salary.replace(0, 1e-5)

    df["bonus_ratio"] = df.get("bonus", 0) / base_salary_safe
    df["stock_ratio"] = df.get("stock_options", 0) / base_salary_safe

    years_exp = df.get("years_experience", pd.Series(np.zeros(len(df)))) + 1
    df["usd_per_year_experience"] = df.get("adjusted_total_usd", 0) / years_exp

    logger.info("✅ Domain-specific features added.")
    save_stage(df, "domain_features")
    return df

def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🔗 Adding interaction features...")
    df = df.copy()

    df["salary_bonus_interaction"] = df.get("base_salary", 0) * df.get("bonus", 0)
    df["exp_salary_interaction"] = df.get("years_experience", 0) * df.get("adjusted_total_usd", 0)

    logger.info("✅ Interaction features added.")
    save_stage(df, "interaction_features")
    return df

def compute_shap_importance(df: pd.DataFrame, target_col: str = "adjusted_total_usd") -> pd.DataFrame:
    logger.info("📈 Computing SHAP feature importance using XGBoost...")
    df = df.copy().dropna(subset=[target_col])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    original_cols = X.columns.tolist()
    safe_cols = [col.replace('[', '(').replace(']', ')').replace('<', 'lt') for col in original_cols]
    X.columns = safe_cols

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap_df = pd.DataFrame({
        'feature': original_cols,
        'shap_importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='shap_importance', ascending=False)

    logger.info("✅ SHAP importance computed.")
    return shap_df

def save_shap_plot(shap_df: pd.DataFrame, plot_path: str = IMPORTANCE_PLOT_PATH, csv_path: str = IMPORTANCE_CSV_PATH):
    logger.info("📊 Saving SHAP importance plot...")
    shap_df_sorted = shap_df.sort_values(by="shap_importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(shap_df_sorted["feature"], shap_df_sorted["shap_importance"], color="skyblue")
    plt.xlabel("SHAP Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Save SHAP values to CSV
    shap_df.to_csv(csv_path, index=False)

    # MLflow logging
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(csv_path)

    logger.info(f"📎 SHAP plot saved: {plot_path}")
    logger.info(f"📄 SHAP CSV saved: {csv_path}")

def full_feature_engineering_pipeline(df: pd.DataFrame, target_col: str = "adjusted_total_usd") -> pd.DataFrame:
    logger.info("🚀 Running full feature engineering pipeline...")
    df = engineer_features(df)
    df = apply_domain_features(df)
    df = add_interaction_terms(df)

    if target_col not in df.columns:
        logger.warning(f"⚠️ Target column '{target_col}' not found. Skipping SHAP importance.")
    else:
        try:
            shap_df = compute_shap_importance(df, target_col)
            save_shap_plot(shap_df)
        except Exception as e:
            logger.warning(f"⚠️ SHAP computation failed: {e}")

    logger.info("✅ Feature engineering pipeline complete.")
    return df
