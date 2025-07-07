import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.utils.logger import setup_logger
from src.data_processing.preprocessor.cleaning import clean_data
from src.data_processing.preprocessor.type_conversion import convert_types
from src.data_processing.preprocessor.missing_imputation import impute_missing
from src.data_processing.preprocessor.outlier_handler import handle_outliers
from src.data_processing.preprocessor.feature_engineering import (
    engineer_features,
    apply_domain_features,
    add_interaction_terms
)
from src.data_processing.preprocessor.encoding import encode_features
from src.data_processing.preprocessor.rare_label_encoder import rare_label_encode
from src.data_processing.preprocessor.target_encoding import target_encode
from src.data_processing.preprocessor.binning import bin_feature
from src.data_processing.preprocessor.clustering import add_clusters

logger = setup_logger(__name__)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

MODEL_NAME = "CatBoost"  # 🔁 Updated model name

def load_latest_production_model(model_name=MODEL_NAME):
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        raise Exception(f"No production versions found for model {model_name}")

    model_uri = f"models:/{model_name}/Production"
    logger.info(f"Loading model from MLflow URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    encoder = joblib.load(os.path.join("artifacts", "encoder.pkl"))
    feature_scaler = joblib.load(os.path.join("artifacts", "feature_scaler.pkl"))
    target_scaler = joblib.load(os.path.join("artifacts", "target_scaler.pkl"))

    return model, encoder, feature_scaler, target_scaler

model, encoder, feature_scaler, target_scaler = load_latest_production_model()

def ensure_columns_exist(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "normalized_role": "",
        "stock_ratio": 0.0,
        "salary_bonus_interaction": 0.0,
        "salary_currency": "",
        "currency": "",
        "job_title": "",
        "years_experience": 0,
        "years_experience_bin": "",
        "company_location": "",
        "company_size": "",
        "employment_type": "",
        "experience_level": "",
        "remote_ratio": 0,
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default
    return df

def preprocess_input(raw_df: pd.DataFrame, target: str = "adjusted_total_usd") -> pd.DataFrame:
    df = raw_df.copy()
    df = ensure_columns_exist(df)

    df = clean_data(df)
    df = convert_types(df)
    df = impute_missing(df)
    df = handle_outliers(df)
    df = engineer_features(df)
    df = apply_domain_features(df)
    df = add_interaction_terms(df)

    for col in ["normalized_role", "company_location"]:
        if col in df.columns:
            df = rare_label_encode(df, col, threshold=0.01)

    if target and target in df.columns and "company_location" in df.columns and "company_size" in df.columns:
        df = target_encode(df, target=target, cat_cols=["company_location", "company_size"])

    if "years_experience" in df.columns:
        df = bin_feature(df, column="years_experience", bins=4)

    categorical_ordered = {
        "experience_level": ["Entry", "Mid", "Senior", "Executive", "Unknown"],
        "employment_type": ["Internship", "Part-Time", "Contract", "Full-Time", "Unknown"]
    }
    for col, categories in categorical_ordered.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories, ordered=True)

    for col in df.select_dtypes(include='category').columns:
        if df[col].isnull().any():
            fill_val = "Unknown"
            if fill_val not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([fill_val])
            df[col] = df[col].fillna(fill_val)

    df = encode_features(df)

    expected_cols = list(feature_scaler.feature_names_in_)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[expected_cols]
    df = pd.DataFrame(feature_scaler.transform(df), columns=expected_cols)
    df = add_clusters(df)

    return df

def predict_df(raw_df: pd.DataFrame) -> pd.Series:
    logger.info("🔮 Running prediction service preprocessing...")
    try:
        X_processed = preprocess_input(raw_df, target="adjusted_total_usd")
    except Exception as e:
        logger.warning(f"⚠️ Preprocessing failed. Retrying without target encoding. Error: {e}")
        X_processed = preprocess_input(raw_df, target=None)

    logger.info(f"✅ Preprocessed shape: {X_processed.shape}")

    preds_scaled = model.predict(X_processed)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)

    try:
        preds_log = target_scaler.inverse_transform(preds_scaled).flatten()
        preds_original = np.expm1(preds_log)
        logger.info("✅ Inverse transform and expm1 applied to predictions.")
    except Exception as e:
        logger.warning(f"⚠️ Could not inverse-transform predictions: {e}")
        preds_original = preds_scaled.flatten()

    return pd.Series(np.round(preds_original, 2), name="adjusted_total_usd")
