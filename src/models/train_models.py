import warnings
warnings.filterwarnings("ignore", message="Tcl_AsyncDelete: async handler deleted by the wrong thread")

import os
import atexit
import shutil
import tempfile
import mlflow
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

from src.utils.logger import setup_logger
from src.utils.metrics import mean_absolute_scaled_error
from src.models.optuna_tuner import optimize_model

_temp_joblib_dir = tempfile.mkdtemp()
os.environ["JOBLIB_TEMP_FOLDER"] = _temp_joblib_dir
atexit.register(lambda: shutil.rmtree(_temp_joblib_dir, ignore_errors=True))

logger = setup_logger(__name__)
client = MlflowClient()
console = Console()

os.makedirs("catboost_logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("drift_reports", exist_ok=True)

NUM_JOBS = 2

MODEL_DICT = {
    "Ridge": Ridge,
    "LightGBM": LGBMRegressor,
    "CatBoost": CatBoostRegressor,
    "XGBoost": XGBRegressor,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
}

TUNING_SPACE = {
    "Ridge": lambda trial: {"alpha": trial.suggest_float("alpha", 0.1, 10.0)},
    "RandomForest": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "n_jobs": NUM_JOBS,
    },
    "GradientBoosting": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
    },
    "XGBoost": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "n_jobs": NUM_JOBS,
    },
    "LightGBM": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 31, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.01),
        "n_jobs": NUM_JOBS,
    },
    "CatBoost": lambda trial: {
        "iterations": trial.suggest_int("iterations", 50, 100, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
        "depth": trial.suggest_int("depth", 3, 6),
        "verbose": 0,
        "train_dir": "./catboost_logs",
    },
}

def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MASE": mean_absolute_scaled_error(y_true, y_pred),
    }

def safe_start_run(run_name=None, nested=False):
    if mlflow.active_run():
        if nested:
            return mlflow.start_run(run_name=run_name, nested=True)
        mlflow.end_run()
    return mlflow.start_run(run_name=run_name, nested=nested)

def train_model(name, ModelClass, params, X, y):
    model = ModelClass(**params)
    model.fit(X, y)
    return model

def train_all_models(df, target="adjusted_total_usd", experiment_name="Model_Training", n_trials=30):
    console.rule("[bold blue]\U0001F680 Starting Model Training")
    mlflow.set_experiment(experiment_name)

    df = df.dropna(subset=[target])
    df[target] = df[target].clip(lower=0)
    df.to_csv("artifacts/historic_reference.csv", index=False)

    X = df.drop(columns=[target])
    y = df[target]

    y_log = np.log1p(y)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_log.values.reshape(-1, 1)).flatten()
    joblib.dump(target_scaler, "artifacts/target_scaler.pkl")

    X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    y_train_true = np.expm1(target_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten())
    y_test_true = np.expm1(target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten())

    results, best_model, best_score = [], None, -np.inf
    best_model_name, best_run_id = None, None

    with safe_start_run(run_name="Experiment_Run") as parent_run:
        model_results = []
        with Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("[green]Training models...", total=len(MODEL_DICT))
            for name, ModelClass in MODEL_DICT.items():
                best_params = optimize_model(ModelClass, TUNING_SPACE[name], X_train, y_train_scaled, n_trials=n_trials)
                model = train_model(name, ModelClass, best_params, X_train, y_train_scaled)

                train_preds_scaled = model.predict(X_train)
                test_preds_scaled = model.predict(X_test)
                train_preds = np.expm1(target_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten())
                test_preds = np.expm1(target_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten())

                train_metrics = evaluate_model(y_train_true, train_preds)
                test_metrics = evaluate_model(y_test_true, test_preds)

                with safe_start_run(run_name=name, nested=True) as run:
                    mlflow.log_params(best_params)
                    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                    mlflow.sklearn.log_model(model, "model")
                    mlflow.set_tags({"model_name": name, "model_type": ModelClass.__name__})

                    model_results.append((name, model, test_metrics, run.info.run_id))
                    results.append({"model": name, **train_metrics, **test_metrics, "run_id": run.info.run_id})

                    if test_metrics["R2"] > best_score:
                        best_model, best_score, best_model_name, best_run_id = model, test_metrics["R2"], name, run.info.run_id

                progress.advance(task)

        # Stacking
        estimators = [(name.lower(), model) for name, model, _, _ in model_results]
        final_estimator = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        stacker = StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=NUM_JOBS, passthrough=True)
        stacker.fit(X_train, y_train_scaled)

        stack_test_preds_scaled = stacker.predict(X_test)
        stack_test_preds = np.expm1(target_scaler.inverse_transform(stack_test_preds_scaled.reshape(-1, 1)).flatten())
        stack_test_metrics = evaluate_model(y_test_true, stack_test_preds)

        with safe_start_run(run_name="StackingEnsemble", nested=True) as run:
            mlflow.log_metrics({f"test_{k}": v for k, v in stack_test_metrics.items()})
            mlflow.sklearn.log_model(stacker, "model")
            mlflow.set_tags({"model_name": "StackingEnsemble", "model_type": "StackingRegressor"})

            results.append({"model": "StackingEnsemble", **stack_test_metrics, "run_id": run.info.run_id})

            if stack_test_metrics["R2"] > best_score:
                best_model, best_score, best_model_name, best_run_id = stacker, stack_test_metrics["R2"], "StackingEnsemble", run.info.run_id

    if best_model_name:
        model_uri = f"runs:/{best_run_id}/model"
        model_version = mlflow.register_model(model_uri, best_model_name)

        all_versions = client.search_model_versions(f"name='{best_model_name}'")
        for v in all_versions:
            if int(v.version) != int(model_version.version):
                client.transition_model_version_stage(name=best_model_name, version=v.version, stage="Archived")

        client.transition_model_version_stage(name=best_model_name, version=model_version.version, stage="Production")
        client.set_registered_model_alias(best_model_name, "champion", model_version.version)

    logger.info("✅ Training completed. Best Test R2: %.4f", best_score)
    return best_model
