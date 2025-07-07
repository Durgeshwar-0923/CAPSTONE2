import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def add_clusters(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features: list[str] = None,
    label_col: str = "cluster_label"
) -> pd.DataFrame:
    """
    Adds cluster labels using KMeans based on numeric features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_clusters (int): Number of clusters.
        features (list[str], optional): Specific features to cluster on. Defaults to all numeric features.
        label_col (str): Name of the new cluster label column.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    logger.info(f"🧩 Adding KMeans cluster-based feature with {n_clusters} clusters...")
    df = df.copy()

    # Select features
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    if not features:
        logger.warning("⚠️ No numeric features available for clustering. Skipping clustering step.")
        df[label_col] = -1
        return df

    # Drop rows with NaNs in cluster features but keep original index
    X = df[features].copy()
    nan_mask = X.isnull().any(axis=1)
    if nan_mask.all():
        logger.warning("⚠️ All rows have NaNs in clustering features. Skipping clustering.")
        df[label_col] = -1
        return df

    X_clean = X[~nan_mask]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Initialize cluster label with -1 for all rows
    df[label_col] = -1

    # Assign cluster labels back to original dataframe rows without NaNs in features
    df.loc[X_clean.index, label_col] = labels

    logger.info(f"✅ Clustering complete. Inertia: {kmeans.inertia_:.2f} | Column added: '{label_col}'")
    return df
