# File: src/models/optuna_tuner.py

import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def optimize_model(
    estimator_class,
    param_space_func,
    X,
    y,
    metric="r2",           # Choose "r2" to maximize R²
    timeout=600,
    n_trials=50,
    cv_splits=3,
    random_state=42,
    early_stopping_rounds=None,
):
    """
    Optimize model hyperparameters with Optuna.

    Args:
        estimator_class: sklearn-like estimator class
        param_space_func: function(trial) -> dict of hyperparameters
        X: feature DataFrame or array
        y: target array
        metric: "r2" or "rmse" (root mean squared error)
        timeout: tuning timeout in seconds
        n_trials: maximum number of trials
        cv_splits: number of CV folds
        random_state: random seed for reproducibility
        early_stopping_rounds: int or None, enable early stopping if supported

    Returns:
        best_params dict
    """

    def objective(trial):
        try:
            params = param_space_func(trial)

            # Add early stopping params if supported and requested
            if early_stopping_rounds is not None:
                if "n_iter_no_change" in estimator_class().get_params().keys():
                    params["n_iter_no_change"] = early_stopping_rounds
                    params["validation_fraction"] = 0.1
                    params["tol"] = 1e-4

            model = estimator_class(**params)
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

            if metric == "r2":
                scores = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
                score_mean = scores.mean()
                logger.info(f"Trial {trial.number} R² mean: {score_mean:.4f}")
                return score_mean  # maximize R2

            # elif metric == "rmse":
            #     # RMSE part commented out as per request
            #     neg_mse_scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
            #     rmse_scores = np.sqrt(-neg_mse_scores)
            #     score_mean = -rmse_scores.mean()  # negate because Optuna maximizes
            #     logger.info(f"Trial {trial.number} RMSE mean: {-score_mean:.4f}")
            #     return score_mean  # maximize negative RMSE = minimize RMSE

            else:
                raise ValueError(f"Unsupported metric: {metric}")

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with exception: {e}")
            return float("-inf")

    logger.info(f"🔧 Starting Optuna tuning for {estimator_class.__name__} with metric={metric}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        timeout=timeout,
        n_trials=n_trials,
        callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))],
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    logger.info(f"✅ Best trial params for {estimator_class.__name__}: {best_params}")

    os.makedirs("outputs", exist_ok=True)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"outputs/optuna_{estimator_class.__name__.lower()}_history.html")
    except Exception as e:
        logger.warning(f"Could not save optimization plot: {e}")

    # Optional: Plot feature importance for tree-based models
    try:
        best_model = estimator_class(**best_params)
        best_model.fit(X, y)
        if hasattr(best_model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            plt.title(f"{estimator_class.__name__} Feature Importances")
            plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
            plt.xlabel("Feature index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(f"outputs/{estimator_class.__name__.lower()}_feature_importances.png")
            plt.close()
    except Exception as e:
        logger.warning(f"Could not plot feature importances: {e}")

    return best_params
