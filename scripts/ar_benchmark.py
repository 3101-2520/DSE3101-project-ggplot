"""
AR Benchmark Model for GDP Growth
--------------------------------
This script implements a simple AR(p) benchmark model for nowcasting GDP growth.
"""

import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scripts.data_preprocessing import load_and_transform_qd

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
    return best_model

def forecast_gdp(model, steps=4):
    forecast = model.forecast(steps=steps)
    print("Next quarter AR nowcast:", forecast.iloc[0])
    print("\nGDP Growth Forecast:")
    print(forecast)
    return forecast

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]
    qd_path = ROOT / "data/2026-02-QD.csv"
    
    # Load and transform quarterly data
    GDP_growth = load_and_transform_qd(qd_path)

    # Fit AR benchmark model
    ar_model = fit_ar_benchmark(GDP_growth)
    print(ar_model.summary())

    # Forecast GDP growth for the next 4 quarters
    forecast_gdp(ar_model, steps=4)
    print("\nAR benchmark complete.")
