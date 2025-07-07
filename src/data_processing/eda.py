import os
import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def clean_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace clearly malformed placeholder strings with NaN.
    Essential cleanup only — no imputation, no column drops.
    """
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.replace(['None', 'none', 'N/A', 'NA', 'null', 'NULL'], pd.NA)
    return df

def run_sweetviz(df: pd.DataFrame, output_path: str = "data/drift_reports/eda_report.html") -> None:
    """
    Generate an automated EDA report using Sweetviz.
    """
    logger.info("🧪 Starting Sweetviz EDA...")
    df = clean_placeholders(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = sv.analyze(df)
    report.show_html(output_path)
    logger.info(f"✅ Sweetviz report saved to: {output_path}")

def eda_summary(df: pd.DataFrame) -> None:
    """
    Display a basic summary of the dataset in the logs.
    """
    logger.info("📊 Running EDA summary...")
    df = clean_placeholders(df)

    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Data types:\n{df.dtypes}")

    # Log summaries separately by dtype for clarity
    numeric_df = df.select_dtypes(include='number')
    object_df = df.select_dtypes(include='object')

    logger.info(f"🧮 Numeric Summary:\n{numeric_df.describe()}")
    logger.info(f"🔤 Categorical Summary:\n{object_df.describe(include='all')}")

def plot_distributions(df: pd.DataFrame, columns=None, save=True, save_dir="reports/plots") -> None:
    """
    Plot histograms with KDE for numeric columns.
    """
    logger.info("📈 Plotting distributions...")
    df = clean_placeholders(df)

    if columns is None:
        columns = df.select_dtypes(include='number').columns

    if save:
        os.makedirs(save_dir, exist_ok=True)

    for col in columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        if save:
            path = os.path.join(save_dir, f"{col}_dist.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot for {col} to {path}")
        else:
            plt.show()
