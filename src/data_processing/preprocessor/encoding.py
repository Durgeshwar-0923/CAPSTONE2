import os
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🔢 Starting feature encoding...")
    df = df.copy()

    # Define full category maps
    full_ordinal_features = {
        "experience_level": ["Junior", "Mid", "Senior", "Lead"],
        "company_size": ["Small", "Medium", "Large"]
    }
    onehot_features = ["employment_type", "remote_ratio"]

    # Filter features that exist in df
    ordinal_features = {
        col: cats for col, cats in full_ordinal_features.items() if col in df.columns
    }
    actual_ohe_features = [col for col in onehot_features if col in df.columns]

    # Fill missing values and set category order
    for col, order in ordinal_features.items():
        df[col] = df[col].fillna(order[1])
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    for col in actual_ohe_features:
        df[col] = df[col].fillna("Unknown")

    # Prepare ColumnTransformer
    transformers = []

    if ordinal_features:
        transformers.append((
            "ord",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            list(ordinal_features.keys())
        ))

    if actual_ohe_features:
        transformers.append((
            "ohe",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            actual_ohe_features
        ))

    remainder_cols = [
        col for col in df.columns
        if col not in ordinal_features and col not in actual_ohe_features
    ]

    transformer = ColumnTransformer(transformers=transformers, remainder="passthrough")
    transformed = transformer.fit_transform(df)

    # ─── Save encoder ───────────────────────────────────────────
    encoder_path = os.path.join(ARTIFACT_DIR, "encoder.pkl")
    joblib.dump(transformer, encoder_path)
    logger.info(f"💾 Saved fitted encoder to → {encoder_path}")

    # ─── Build final column names ──────────────────────────────
    final_columns = []

    if "ord" in transformer.named_transformers_:
        final_columns.extend(ordinal_features.keys())

    if "ohe" in transformer.named_transformers_:
        ohe = transformer.named_transformers_["ohe"]
        final_columns.extend(ohe.get_feature_names_out(actual_ohe_features))

    final_columns.extend(remainder_cols)

    # Create encoded DataFrame
    encoded_df = pd.DataFrame(transformed, columns=final_columns)

    # Ensure all columns are numeric (important for VIF, SHAP, etc.)
    encoded_df = encoded_df.apply(pd.to_numeric, errors="coerce")

    logger.info(f"✅ Encoding complete. Output shape: {encoded_df.shape}")
    return encoded_df
