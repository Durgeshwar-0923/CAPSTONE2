import os
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts column types for consistent processing:
    - Object → numeric (if possible), else category
    - Dates inferred and converted
    - Boolean strings converted to True/False

    Returns:
        pd.DataFrame: DataFrame with updated types
    """
    logger.info("🧾 Converting column data types...")
    df = df.copy()
    type_conversion_log = {}

    for col in df.columns:
        original_dtype = str(df[col].dtype)
        new_type = None

        # Handle boolean-like strings
        if df[col].dropna().astype(str).str.lower().isin(["true", "false", "yes", "no"]).all():
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False, "yes": True, "no": False})
            new_type = "bool"
            logger.info(f"🔘 Converted {col} to boolean.")

        else:
            # Try numeric conversion
            numeric_converted = pd.to_numeric(df[col], errors='coerce')
            if numeric_converted.notna().sum() > 0 and numeric_converted.isna().sum() < df[col].isna().sum() + 10:
                df[col] = numeric_converted
                new_type = "numeric"
                logger.info(f"🔢 Converted {col} to numeric.")

            else:
                # Try date parsing
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > 0:
                        df[col] = parsed
                        new_type = "datetime"
                        logger.info(f"📅 Converted {col} to datetime.")
                except Exception:
                    pass

        # Fallback to category if no change happened
        if new_type is None and df[col].dtype == "object":
            df[col] = df[col].astype("category")
            new_type = "category"
            logger.info(f"🔤 Converted {col} to category.")

        type_conversion_log[col] = {
            "original_dtype": original_dtype,
            "converted_dtype": str(df[col].dtype)
        }

    # Save artifact
    conversion_log_path = os.path.join(ARTIFACT_DIR, "type_conversions.csv")
    pd.DataFrame(type_conversion_log).T.to_csv(conversion_log_path)
    logger.info(f"💾 Saved type conversion log to {conversion_log_path}")

    logger.info("✅ Type conversion complete.")
    return df
