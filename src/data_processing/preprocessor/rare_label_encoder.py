import os
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def rare_label_encode(df: pd.DataFrame, column: str, threshold: float = 0.01, new_col: str = None) -> pd.DataFrame:
    """
    Encodes rare categories in a given column as 'Rare'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to encode.
        threshold (float): Frequency threshold below which a category is considered rare.
        new_col (str): Optional name for the encoded column. If None, replaces the original column.

    Returns:
        pd.DataFrame: DataFrame with rare categories encoded.
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df[column] = df[column].astype(str)
    freq = df[column].value_counts(normalize=True)
    rare_labels = freq[freq < threshold].index.tolist()

    encoded_col = new_col if new_col else column
    df[encoded_col] = df[column].apply(lambda x: "Rare" if x in rare_labels else x)

    logger.info(f"🏷️ Encoded rare labels in '{column}' → {'new column ' + encoded_col if new_col else 'in place'}")
    logger.info(f"🔹 Rare labels (<{threshold*100:.1f}%): {len(rare_labels)} found.")

    # Save artifact
    artifact_path = os.path.join(ARTIFACT_DIR, f"rare_labels_{column}.txt")
    with open(artifact_path, "w") as f:
        f.write("\n".join(rare_labels))
    logger.info(f"💾 Saved rare labels to {artifact_path}")

    return df
