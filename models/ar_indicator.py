
from config import *


def fit_ar_models(monthly_data, selected_names, max_lag=12, verbose=VERBOSE):
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
        
        if verbose:
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
            if verbose:
                print(f"✓ AR({best_p}) fitted for {name}")
        else:
            if verbose:
                print(f"✗ No AR model could be fitted for {name}. Using mean fallback.")
            # Fallback: always predict the series mean
            class MeanForecast:
                def forecast(self, steps):
                    return np.full(steps, series.mean())
            ar_models[name] = MeanForecast()
    
    return ar_models

def fill_ragged_edge(monthly_data, ar_models, selected_names, verbose=VERBOSE):
    """
    Fills the ragged edge of the monthly indicators using the fitted AR models.

    For each selected indicator, the corresponding AR model is used to forecast values for
    the next 3 months (1 quarter) beyond the last available date. The original monthly data
    is extended with these forecasts, resulting in a DataFrame that has no missing values
    for the selected indicators up to 3 months after the last date.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Original monthly DataFrame with date index and all indicators.
    ar_models : dict
        Dictionary of fitted AR models for each selected indicator (output of `fit_ar_models`).
    selected_names : list of str
        Names of the indicators for which forecasts should be generated.

    Returns
    -------
    pd.DataFrame
        Extended monthly DataFrame with forecasts filled in for the ragged edge.
    """
    filled_data = monthly_data.copy()
    
    for name in selected_names:
        series = filled_data[name].copy()
        # skip if no missing values at the end (i.e., no ragged edge)
        if not series.isna().any():
            if verbose:
                print(f"No ragged edge for {name}, skipping AR forecast.")
            continue
        last_valid_index = series.last_valid_index()
        # last observation date for this series
        if last_valid_index is None:
            if verbose:
                print(f"Warning: {name} has no valid data to fit AR model. Skipping.")
            continue
        # all missig values after last_valid_index
        tail = series.loc[last_valid_index:].copy()
        n_missing = tail.isna().sum()
        if n_missing == 0:
            if verbose:
                print(f"No missing values at the end of {name}, skipping AR forecast.")
            continue
        forecasts = ar_models[name].forecast(steps=n_missing)

        # fill the missing values
        missing_idx = series.loc[last_valid_index:].index[series.loc[last_valid_index:].isna()]
        filled_data.loc[missing_idx, name] = np.asarray(forecasts)
        if verbose:
            print(f"Filled {n_missing} missing values for {name} using AR forecast.")
    return filled_data
