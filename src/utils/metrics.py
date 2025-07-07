# src/utils/metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error

def mean_absolute_scaled_error(y_true, y_pred, y_train_mean=None):
    """
    Computes the Mean Absolute Scaled Error (MASE).
    If `y_train_mean` is not provided, it uses the in-sample naive forecast.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = y_true.shape[0]

    if y_train_mean is None:
        d = np.abs(np.diff(y_true)).mean()
    else:
        d = y_train_mean

    errors = np.abs(y_true - y_pred)
    return errors.mean() / d if d != 0 else np.nan