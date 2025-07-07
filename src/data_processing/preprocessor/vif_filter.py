import os
import pandas as pd
import numpy as np
import mlflow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ARTIFACT_DIR = "outputs/importance"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def vif_filter(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Removes features with high multicollinearity using VIF and logs final VIFs.
    Missing and infinite values are imputed to preserve all rows.

    Args:
        df (pd.DataFrame): Input numeric DataFrame.
        threshold (float): VIF threshold.

    Returns:
        pd.DataFrame: Filtered numeric DataFrame with all original rows.
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include='number').copy()

    if numeric_df.empty or numeric_df.shape[1] < 2:
        logger.warning("⚠️ Not enough numeric features. Skipping VIF filtering.")
        return numeric_df

    # Handle missing and infinite values by imputation
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    numeric_df[numeric_df.columns] = imputer.fit_transform(numeric_df)
    logger.info("🛠️ Imputed missing/infinite numeric features before VIF filtering")

    logger.info(f"🔍 Starting VIF filtering with threshold={threshold}...")
    vif_history = []
    iteration = 0

    while True:
        vif_data = pd.DataFrame({
            'feature': numeric_df.columns,
            'VIF': [variance_inflation_factor(numeric_df.values, i)
                    for i in range(numeric_df.shape[1])]
        })
        vif_history.append(vif_data.copy())

        max_vif = vif_data['VIF'].max()
        if max_vif > threshold:
            drop_feature = vif_data.sort_values('VIF', ascending=False).iloc[0]['feature']
            logger.info(f"🚫 Dropping '{drop_feature}' with VIF={max_vif:.2f}")
            numeric_df.drop(columns=[drop_feature], inplace=True)
            iteration += 1
            if numeric_df.shape[1] < 2:
                logger.warning("⚠️ Only one feature left. Stopping VIF filtering.")
                break
        else:
            break

    logger.info(f"✅ VIF filtering complete. Retained {numeric_df.shape[1]} features after {iteration} iterations.")

    # Save final VIF scores
    final_vif = vif_history[-1]
    vif_csv_path = os.path.join(ARTIFACT_DIR, "vif_scores.csv")
    final_vif.to_csv(vif_csv_path, index=False)
    mlflow.log_artifact(vif_csv_path, artifact_path="importance")
    logger.info(f"📄 Final VIF scores saved and logged to {vif_csv_path}")

    return numeric_df
