"""
Nowcasting project – Data preprocessing and feature selection
------------------------------------------------------------
This script:
1. Loads FRED‑MD (monthly) and FRED‑QD (quarterly) CSV files.
2. Applies the recommended t‑code transformations to achieve stationarity.
3. Aggregates monthly indicators to quarterly frequency.
4. Merges with quarterly GDP growth.
5. Uses `hdmpy.rlasso` to select the most relevant monthly indicators.
6. (Placeholders) Fits bridge equation and AR(p) models for later nowcasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdmpy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_and_transform_md(filepath):
    """
    Load FRED‑MD CSV, extract transformation codes, apply them,
    and return a DataFrame of stationary monthly series (index = date).
    """
    md = pd.read_csv(filepath)
    # First row contains t‑codes
    tcodes = md.iloc[0]
    md = md.iloc[1:].copy()
    
    # Parse dates
    md['sasdate'] = pd.to_datetime(md['sasdate'], format='%m/%d/%Y')
    md.set_index('sasdate', inplace=True)
    
    # Apply transformations to every column
    transformed_list = []
    for col in md.columns:
        code = int(tcodes[col])
        s = transform_series(md[col].astype(float), code)
        s.name = col
        transformed_list.append(s)
    
    md_trans = pd.concat(transformed_list, axis=1)
    md_trans.dropna(inplace=True)          # remove rows with NaN from differencing
    print("FRED‑MD transformation complete. Shape:", md_trans.shape)
    return md_trans


def load_and_transform_qd(filepath, gdp_col='GDPC1'):
    """
    Load FRED‑QD CSV, extract transformation codes, apply them to the GDP series,
    and return a Series of quarterly GDP growth (index = period).
    """
    qd = pd.read_csv(filepath)
    tcodes_qd = qd.iloc[0]
    qd = qd.iloc[1:].copy()
    
    # Parse dates (format may vary; adjust if needed)
    qd['sasdate'] = pd.to_datetime(qd['sasdate'], format='%m/%d/%Y', errors='coerce')
    qd.dropna(subset=['sasdate'], inplace=True)
    qd.set_index('sasdate', inplace=True)
    
    # Transform the GDP column
    code_gdp = int(tcodes_qd[gdp_col])
    gdp_raw = qd[gdp_col].astype(float)
    gdp_trans = transform_series(gdp_raw, code_gdp)
    gdp_trans.name = 'GDP_growth'
    gdp_trans.dropna(inplace=True)
    
    # Convert index to quarterly period
    gdp_trans.index = gdp_trans.index.to_period('Q')
    print("FRED‑QD transformation complete. Length:", len(gdp_trans))
    return gdp_trans


def transform_series(series, code):
    """
    Apply a given transformation code to a pandas Series.
    Codes follow McCracken & Ng (2016):
        1 : no transformation
        2 : first difference
        3 : second difference
        4 : log
        5 : log first difference
        6 : log second difference
    """
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()
    else:
        return series   # fallback (should not happen)


def aggregate_to_quarterly(monthly_df):
    """
    Convert monthly DataFrame (index = date) to quarterly by averaging.
    Returns a DataFrame with a PeriodIndex (quarters).
    """
    # Add a quarter column
    monthly_df = monthly_df.copy()
    monthly_df['quarter'] = monthly_df.index.to_period('Q')
    # Group by quarter and take mean
    quarterly = monthly_df.groupby('quarter').mean()
    return quarterly


def merge_data(monthly_q, gdp_series):
    """
    Merge quarterly aggregates of monthly indicators with GDP growth.
    Returns a DataFrame (rows = quarters) and separate X, y arrays.
    """
    data = monthly_q.join(gdp_series, how='inner').dropna()
    X = data.drop(columns=['GDP_growth']).values
    y = data['GDP_growth'].values
    print("Quarterly dataset shape:", data.shape)
    print("Number of predictors:", X.shape[1])
    return data, X, y


def select_features_rlasso(X, y, feature_names, threshold=1e-6):
    """
    Run rlasso from hdmpy, extract non‑zero coefficients,
    and return the names of selected variables.
    """
    rlasso_result = hdmpy.rlasso(X, y, post=True)
    
    # Coefficients are stored in rlasso_result.est['coefficients'] as a DataFrame
    coefs_df = rlasso_result.est['coefficients']
    coefs_all = coefs_df.values.flatten()
    
    # If the length includes intercept (should be n_features + 1), drop it
    if len(coefs_all) == X.shape[1] + 1:
        coefs = coefs_all[1:]
    else:
        coefs = coefs_all
    
    selected_idx = np.where(np.abs(coefs) > threshold)[0]
    selected_names = feature_names[selected_idx]
    print("Selected variables:", list(selected_names))
    print("Number of selected variables:", len(selected_names))
    return selected_names


def fit_bridge_equation(data, selected_names):
    """
    Fit OLS bridge equation: GDP_growth ~ selected indicators (quarterly aggregates).
    Returns the fitted model and a dictionary of coefficients.
    """
    X_sel = data[selected_names]
    X_sel = sm.add_constant(X_sel)
    y = data['GDP_growth']
    model = sm.OLS(y, X_sel).fit()
    print(model.summary())
    
    # Store coefficients
    coef_dict = {'intercept': model.params['const']}
    for name in selected_names:
        coef_dict[name] = model.params[name]
    return model, coef_dict


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


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # File paths (adjust if needed)
    md_path = "data/2026-02-MD.csv"
    qd_path = "data/2026-02-QD.csv"
    
    # Step 1: Load and transform monthly data
    MD_trans = load_and_transform_md(md_path)
    
    # Step 2: Load and transform quarterly GDP
    GDP_growth = load_and_transform_qd(qd_path, gdp_col='GDPC1')
    
    # Step 3: Aggregate monthly indicators to quarterly
    monthly_q = aggregate_to_quarterly(MD_trans)
    
    # Step 4: Merge with GDP growth
    data, X, y = merge_data(monthly_q, GDP_growth)
    
    # Step 5: Feature selection with rlasso
    feature_names = data.drop(columns=['GDP_growth']).columns
    selected = select_features_rlasso(X, y, feature_names)
    
    # Step 6: Fit bridge equation (OLS) using selected variables
    bridge_model, bridge_coefs = fit_bridge_equation(data, selected)
    
    # Step 7: Fit AR(p) models for each selected indicator (for ragged‑edge forecasting)
    ar_models = fit_ar_models(MD_trans, selected, max_lag=12)
    
    print("\nAll preprocessing and model fitting complete.")
    print("You can now use the selected variables, bridge coefficients, and AR models for nowcasting.")


















