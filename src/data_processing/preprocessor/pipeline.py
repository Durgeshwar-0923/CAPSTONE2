import os
import pandas as pd
import numpy as np
import mlflow
import joblib

from sklearn.impute import SimpleImputer
from src.data_processing.preprocessor.cleaning import clean_data
from src.data_processing.preprocessor.type_conversion import convert_types
from src.data_processing.preprocessor.missing_imputation import impute_missing
from src.data_processing.preprocessor.outlier_handler import handle_outliers
from src.data_processing.preprocessor.feature_engineering import (
    engineer_features,
    apply_domain_features,
    add_interaction_terms,
    compute_shap_importance,
    save_shap_plot,
)
from src.data_processing.preprocessor.encoding import encode_features
from src.data_processing.preprocessor.scaling import scale_features
from src.data_processing.preprocessor.feature_selection import select_features
from src.data_processing.preprocessor.vif_filter import vif_filter
from src.data_processing.preprocessor.clustering import add_clusters
from src.data_processing.preprocessor.rare_label_encoder import rare_label_encode
from src.data_processing.preprocessor.target_encoding import target_encode
from src.data_processing.preprocessor.binning import bin_feature
from src.utils.logger import setup_logger
from src.config.config import MLOpsConfig

logger = setup_logger(__name__)
mlflow.set_tracking_uri(MLOpsConfig().mlflow_tracking_uri)
mlflow.set_experiment("preprocessing")

OUTPUT_DIRS = [
    "outputs/plots",
    "outputs/distributions",
    "outputs/importance",
    "outputs/stages",
    "artifacts",
    "data/processed",
    "reports"
]
for directory in OUTPUT_DIRS:
    os.makedirs(directory, exist_ok=True)


def log_rows_removed(prev_len: int, curr_len: int, step_name: str):
    removed = prev_len - curr_len
    logger.info(f"🔻 Rows removed at '{step_name}': {removed}. Remaining: {curr_len}")


def run_preprocessing_pipeline(df: pd.DataFrame, target_col: str = "adjusted_total_usd") -> tuple:
    logger.info(f"Initial rows: {len(df)}")
    prev_len = len(df)

    df = clean_data(df)
    log_rows_removed(prev_len, len(df), "clean_data")
    prev_len = len(df)

    df = convert_types(df)
    log_rows_removed(prev_len, len(df), "convert_types")
    prev_len = len(df)

    df = impute_missing(df)
    log_rows_removed(prev_len, len(df), "impute_missing")
    prev_len = len(df)

    df = handle_outliers(df)
    log_rows_removed(prev_len, len(df), "handle_outliers")
    prev_len = len(df)

    df = engineer_features(df)
    log_rows_removed(prev_len, len(df), "engineer_features")
    prev_len = len(df)
    df.to_csv("outputs/stages/feature_engineering.csv", index=False)

    df = apply_domain_features(df)
    log_rows_removed(prev_len, len(df), "apply_domain_features")
    prev_len = len(df)
    df.to_csv("outputs/stages/domain_features.csv", index=False)

    df = add_interaction_terms(df)
    log_rows_removed(prev_len, len(df), "add_interaction_terms")
    prev_len = len(df)
    df.to_csv("outputs/stages/interaction_features.csv", index=False)

    for col in ["normalized_role", "company_location"]:
        prev_len = len(df)
        df = rare_label_encode(df, col, threshold=0.01)
        log_rows_removed(prev_len, len(df), f"rare_label_encode({col})")

    prev_len = len(df)
    df = target_encode(df, target=target_col, cat_cols=["company_location", "company_size"])
    log_rows_removed(prev_len, len(df), "target_encode")

    prev_len = len(df)
    df = bin_feature(df, column="years_experience", bins=4)
    log_rows_removed(prev_len, len(df), "bin_feature(years_experience)")

    prev_len = len(df)
    df = encode_features(df)
    log_rows_removed(prev_len, len(df), "encode_features")

    prev_len = len(df)
    df, feature_scaler = scale_features(df, return_scaler=True)
    log_rows_removed(prev_len, len(df), "scale_features")

    prev_len = len(df)
    df = select_features(df)
    log_rows_removed(prev_len, len(df), "select_features")
    df.to_csv("outputs/stages/feature_selection.csv", index=False)

    # ─── VIF Filtering & Clustering ─────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Impute missing/infinite numeric values (preserving all rows)
    imputer = SimpleImputer(strategy="median")
    df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols, index=df.index)

    logger.info("🛠️ Imputed missing/infinite numeric features before VIF filtering")

    # Apply VIF filter on numeric columns only (no row drops)
    filtered_numeric = vif_filter(df_numeric_imputed, threshold=5.0)
    vif_dropped = list(set(numeric_cols) - set(filtered_numeric.columns.tolist()))

    # Combine filtered numeric with non-numeric columns, preserving all rows
    non_numeric = df.drop(columns=numeric_cols)
    df_filtered = pd.concat([filtered_numeric, non_numeric], axis=1)

    # Add clusters (clustering will handle missing values internally or skip)
    df_final = add_clusters(df_filtered)

    logger.info(f"✅ VIF filtering + clustering complete. Dropped features: {vif_dropped}")
    logger.info("✅ Preprocessing pipeline completed.")
    return df_final, vif_dropped, feature_scaler


def run_with_shap(df: pd.DataFrame, target: str = "adjusted_total_usd"):
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="Preprocessing Pipeline"):
        logger.info("🚀 Running full preprocessing pipeline with SHAP + MLflow...")

        original_df = df.copy()
        processed_df, vif_dropped, feature_scaler = run_preprocessing_pipeline(df, target_col=target)

        mlflow.log_param("target_column", target)
        mlflow.log_metric("num_records", len(processed_df))
        mlflow.log_metric("num_features", processed_df.shape[1])
        mlflow.set_tag("stage", "preprocessing")
        mlflow.set_tag("vif_features_removed", ",".join(vif_dropped) if vif_dropped else "None")

        # ─── SHAP Analysis ──────────────────────────────────────────────────────
        shap_df = None
        if target in original_df.columns:
            try:
                shap_df = compute_shap_importance(processed_df, target_col=target)
                save_shap_plot(shap_df)

                shap_csv = "outputs/importance/shap_feature_importance.csv"
                shap_img = "outputs/importance/shap_feature_importance.png"
                if os.path.exists(shap_csv):
                    mlflow.log_artifact(shap_csv)
                if os.path.exists(shap_img):
                    mlflow.log_artifact(shap_img)

                logger.info("📊 SHAP importance logged.")
            except Exception as e:
                logger.warning(f"⚠️ SHAP computation failed: {e}")
        else:
            logger.warning(f"⚠️ Target column '{target}' not found. Skipping SHAP.")

        # ─── Log Intermediate Stages ────────────────────────────────────────────
        for stage_file in [
            "feature_engineering.csv",
            "domain_features.csv",
            "interaction_features.csv",
            "feature_selection.csv"
        ]:
            path = os.path.join("outputs", "stages", stage_file)
            if os.path.exists(path):
                mlflow.log_artifact(path)

        # ─── Save Final Preprocessed Data ───────────────────────────────────────
        final_csv = os.path.join("data", "processed", "processed_output.csv")
        processed_df.to_csv(final_csv, index=False)
        mlflow.log_artifact(final_csv)

        if shap_df is not None:
            shap_path = os.path.join("reports", "shap_output.csv")
            shap_df.to_csv(shap_path, index=False)
            mlflow.log_artifact(shap_path)

        # ─── Save Scalers ───────────────────────────────────────────────────────
        scaler_path = "artifacts/feature_scaler.pkl"
        joblib.dump(feature_scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # ─── Save Preprocessing Bundle (Placeholder) ────────────────────────────
        preprocessor_path = "artifacts/preprocessing_bundle.pkl"
        joblib.dump(feature_scaler, preprocessor_path)
        mlflow.log_artifact(preprocessor_path)
        logger.info("📦 Saved preprocessing_bundle.pkl to artifacts.")

        logger.info("📦 MLflow tracking complete.")
        return processed_df, shap_df
