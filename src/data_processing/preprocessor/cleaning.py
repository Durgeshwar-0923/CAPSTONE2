import os
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

PLACEHOLDERS = ["", "NA", "N/A", "null", "Null", "none", "None", "-", "--"]

ROLE_MAPPING = {
    'data analyst': 'Data Analyst',
    'data scientist': 'Data Scientist',
    'data scienist': 'Data Scientist',
    'data scntist': 'Data Scientist',
    'dt scientist': 'Data Scientist',

    'ml enginer': 'ML Engineer',
    'ml engr': 'ML Engineer',
    'machine learning engr': 'ML Engineer',
    'machine learning engineer': 'ML Engineer',

    'research scientist': 'Research Scientist',

    'software engr': 'Software Engineer',
    'softwre engineer': 'Software Engineer',
    'sofware engneer': 'Software Engineer',
    'software engineer': 'Software Engineer',

    'devops engineer': 'DevOps Engineer'
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🧹 Starting data cleaning...")
    df = df.copy()

    df.replace(PLACEHOLDERS, np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if "job_title" in df.columns:
        logger.info("🔧 Normalizing job titles...")
        df["job_title"] = df["job_title"].astype(str).str.strip().str.title()
        df["normalized_role"] = (
            df["job_title"]
            .str.lower()
            .map(ROLE_MAPPING)
            .fillna(df["job_title"])
        )

    for cat_col in ["experience_level", "employment_type", "company_size"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip().str.title()

    # Save cleaned data artifact
    cleaned_path = os.path.join(ARTIFACT_DIR, "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    logger.info(f"💾 Saved cleaned data to {cleaned_path}")

    logger.info("✅ Data cleaning complete.")
    return df
