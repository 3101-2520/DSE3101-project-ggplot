
from scripts.config import *


def fit_ar_models(monthly_data, selected_names, max_lag=12):
    """
    Fits autoregressive (AR) models for each selected monthly indicator.

    For each variable in `selected_names`, an AR(p) model with p from 1 to `max_lag` is estimated
    on the series after dropping missing values and resetting the index to integer positions
    (to avoid date‑related issues). The model with the lowest AIC is retained. If no model can
    be fitted (e.g., due to estimation errors), a simple fallback forecast that returns the
    series mean is used.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        DataFrame with monthly data (index = date) containing all potential indicators.
    selected_names : list of str
        Names of the indicators for which AR models should be fitted.
    max_lag : int, optional
        Maximum number of lags to consider (default 12). The actual maximum lag per series
        is limited to half its length to avoid overfitting.

    Returns
    -------
    dict
        A dictionary with keys = indicator names, values = fitted model objects. Each model
        has a `forecast(steps)` method that returns an array of forecasted values.
    """
    ar_models = {}
    for name in selected_names:
        series = monthly_data[name].dropna().copy()
        
        # Reset index to integer positions (0,1,2,…) – eliminates date complications
        series = series.reset_index(drop=True)
        
        print(f"\nAttempting AR for {name}: length = {len(series)}")
        
        best_aic = np.inf
        best_model = None
        best_p = None
        
        for p in range(1, min(max_lag, len(series)//2)):  # avoid p too large
            try:
                model = AutoReg(series, lags=p).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_p = p
            except Exception as e:
                # Print the error for debugging (optional)
                print(f"   AR({p}) failed: {e}")
                continue
        
        if best_model is not None:
            ar_models[name] = best_model
            print(f"✓ AR({best_p}) fitted for {name}")
        else:
            print(f"✗ No AR model could be fitted for {name}. Using mean fallback.")
            # Fallback: always predict the series mean
            class MeanForecast:
                def forecast(self, steps):
                    return np.full(steps, series.mean())
            ar_models[name] = MeanForecast()
    
    return ar_models
