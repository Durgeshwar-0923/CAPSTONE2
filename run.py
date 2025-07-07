# # # run_pipeline.py

# import pandas as pd
# import mlflow
# from sklearn.impute import SimpleImputer

# from src.config.config import Paths, DatabaseConfig, MLOpsConfig
# from src.data_ingestion.data_loader import DataLoader
# from src.data_processing.preprocessor.pipeline import run_with_shap
# from src.models.train_models import train_all_models
# from src.utils.logger import setup_logger

# logger = setup_logger(__name__)

# # MLflow experiment names
# PREPROCESS_EXP = "Preprocessing"
# TRAINING_EXP = "Model_Training"

# def main():
#     # 1️⃣ Load raw data
#     loader = DataLoader(DatabaseConfig())
#     df = loader.load_data()

#     # 2️⃣ Set MLflow tracking URI
#     mlflow.set_tracking_uri(MLOpsConfig().mlflow_tracking_uri)

#     # 3️⃣ Define paths for cached outputs
#     processed_path = Paths.PROCESSED_DATA / "processed_output.csv"
#     shap_path = Paths.REPORTS / "shap_output.csv"

#     # 4️⃣ Load cached if available, otherwise run preprocessing
#     if processed_path.exists():
#         logger.info("📂 Loading previously saved processed data...")
#         processed_df = pd.read_csv(processed_path)
#         shap_df = pd.read_csv(shap_path) if shap_path.exists() else None
#     else:
#         mlflow.set_experiment(PREPROCESS_EXP)
#         with mlflow.start_run(run_name="Preprocessing"):
#             processed_df, shap_df = run_with_shap(df, target="adjusted_total_usd")

#             processed_df = processed_df.replace([float('inf'), float('-inf')], pd.NA)
#             if processed_df.isnull().values.any():
#                 logger.warning("⚠️ Missing values detected post-pipeline. Applying final imputation.")
#                 imputer = SimpleImputer(strategy="mean")
#                 imputed_array = imputer.fit_transform(processed_df)
#                 processed_df = pd.DataFrame(imputed_array, columns=processed_df.columns)

#             processed_df.to_csv(processed_path, index=False)
#             logger.info(f"📁 Saved processed data to {processed_path}")
#             mlflow.log_artifact(str(processed_path))

#             if shap_df is not None:
#                 shap_df.to_csv(shap_path, index=False)
#                 logger.info(f"📊 Saved SHAP results to {shap_path}")
#                 mlflow.log_artifact(str(shap_path))

#     # 5️⃣ Train models
#     mlflow.set_experiment(TRAINING_EXP)
#     best_model = train_all_models(
#         df=processed_df,
#         target="adjusted_total_usd",
#         experiment_name=TRAINING_EXP,
#         n_trials=2  # 🔁 You can increase for better tuning
#     )

#     logger.info("✅ Full pipeline (preprocessing + training) completed.")

# if __name__ == "__main__":
#     main()
# run_pipeline.py

# Only used to launch Flask app from this script
from src.api.app import app

if __name__ == "__main__":
    app.run(debug=True,port=8000)
