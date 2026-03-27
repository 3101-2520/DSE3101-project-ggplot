"""
AR Benchmark Model for GDP Growth
--------------------------------
This script implements a simple AR(p) benchmark model for nowcasting GDP growth.
"""
from config import *
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data_preprocessing import load_and_transform_qd

# ----------------------------------------------------------------------
# AR model functions
# ----------------------------------------------------------------------

def fit_ar_benchmark(series, max_lag = 8):
    best_aic = np.inf
    best_model = None
    best_p = None
    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(series, lags=p).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_p = p
        except:
            continue
    print(f"Selected AR({best_p}) model with AIC: {best_aic:.2f}")
    return best_model, best_p

def run_ar_benchmark(data, test_size=8, target_col = "GDP_growth", max_lag = 8, verbose = VERBOSE):
    """
    Rolling-window AR benchmark evaluation.
    """
    results = []
    total_obs = len(data)
    train_size = total_obs - test_size

    for i in range(test_size):
        train_end_index = train_size - 1 + i
        forecast_index = train_size + i

        train_data = data.iloc[i:train_end_index + 1].copy()
        forecast_quarter = data.index[forecast_index]
        if verbose:
             print(f"\nRolling step {i+1}/{test_size}")
             print(f"Training window: {train_data.index.min()} to {train_data.index.max()}")
             print(f"Forecast quarter: {forecast_quarter}")

        y_train = train_data[target_col].dropna()
        model, best_p = fit_ar_benchmark(y_train, max_lag=max_lag)
        y_pred = model.forecast(steps=1).iloc[0]
        y_actual = data.loc[forecast_quarter, target_col]
        results.append({
            "quarter": forecast_quarter,
            "actual": y_actual,
            "predicted": y_pred
        })
    
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df['error'] = results_df['actual'] - results_df['predicted']
        rmse = np.sqrt((results_df['error'] ** 2).mean())
        mae = np.mean(np.abs(results_df['error']))
        directional_acc = np.mean(np.sign(results_df['actual']) == np.sign(results_df['predicted']))

        print("\nAR Benchmark Results:")
        print(results_df)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Directional Accuracy (Success Ratio): {directional_acc:.3f}")
    else:
        print("No AR benchmark forecasts were generated.")
    return results_df