import numpy as np
import pandas as pd

from models.bridge_model import fit_bridge_model
from models.ar_benchmark import fit_ar_benchmark, predict_ar_benchmark


def compute_metrics(y_true, y_pred):
    """
    Compute forecast evaluation metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # Directional accuracy: did we predict the sign correctly
    directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Directional_Accuracy": directional_accuracy
    }


def expanding_window_bridge(data, feature_cols, target_col="GDP_growth", start_train=40):
    """
    Expanding window evaluation for bridge model.

    Parameters
    ----------
    data : pd.DataFrame
        Quarterly dataset containing predictors and target.
    feature_cols : list
        Final selected bridge-model features.
    target_col : str
        Target variable column name.
    start_train : int
        Initial number of quarters used for first training window.

    Returns
    -------
    pd.DataFrame
        DataFrame with actual and predicted values by quarter.
    """
    results = []

    # Keep only needed columns, remove invalid rows for evaluation
    eval_data = data[feature_cols + [target_col]].copy()
    eval_data = eval_data.replace([np.inf, -np.inf], np.nan).dropna()

    for i in range(start_train, len(eval_data)):
        train = eval_data.iloc[:i]
        test = eval_data.iloc[i:i+1]

        model, _ = fit_bridge_model(train, feature_cols, target_col=target_col)

        X_test = test[feature_cols].copy()
        X_test = np.column_stack([np.ones(len(X_test)), X_test.values])

        pred = float(model.predict(X_test)[0])
        actual = float(test[target_col].iloc[0])

        results.append({
            "quarter": test.index[0],
            "actual": actual,
            "predicted": pred
        })

    return pd.DataFrame(results)


def expanding_window_ar(data, target_col="GDP_growth", lags=2, start_train=40):
    """
    Expanding window evaluation for AR benchmark model.
    """
    results = []

    eval_data = data[[target_col]].copy().dropna()

    for i in range(start_train, len(eval_data)):
        train = eval_data.iloc[:i]
        test = eval_data.iloc[i:i+1]

        model = fit_ar_benchmark(train[target_col], lags=lags)
        pred = predict_ar_benchmark(model)
        actual = float(test[target_col].iloc[0])

        results.append({
            "quarter": test.index[0],
            "actual": actual,
            "predicted": pred
        })

    return pd.DataFrame(results)