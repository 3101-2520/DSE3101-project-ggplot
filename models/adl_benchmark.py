from config import *
import pandas as pd
import numpy as np
import statsmodels.api as sm

def prepare_adl_data(data, target_col = "GDP_growth"):
    """
    Prepare data for ADL benchmark: create lagged predictors and target variable.
    """
    adl_data = data.copy()
    adl_data['GDP_growth_lag1'] = adl_data[target_col].shift(1)
    adl_data['GDP_growth_lag2'] = adl_data[target_col].shift(2)
    adl_data["BAA - AAA"] = adl_data["BAA"] - adl_data["AAA"]
    return adl_data

def fit_adl_benchmark(data, target_col = "GDP_growth", verbose = VERBOSE):
    """
    Fit a simple ADL benchmark:
    GDP_growth ~ GDP_growth_lag1 + GDP_growth_lag2 + (BAA - AAA) + UNRATE + HOUST
    """
    model_data = prepare_adl_data(data, target_col=target_col)
    required_cols = [target_col, 'GDP_growth_lag1', 'GDP_growth_lag2', "BAA - AAA", "UNRATE", "HOUST"]
    model_data = model_data[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = model_data[['GDP_growth_lag1', 'GDP_growth_lag2', "BAA - AAA", "UNRATE", "HOUST"]]
    X = sm.add_constant(X)
    y = model_data[target_col]
    model = sm.OLS(y, X).fit()
    if VERBOSE:
        print(model.summary())
    return model

def run_adl_benchmark(data, test_size=8, target_col = "GDP_growth", verbose = VERBOSE):
    """
    Rolling-window ADL benchmark evaluation.
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

        adl_model = fit_adl_benchmark(train_data, target_col=target_col)
        # prepare forecast row
        full_adl_model = prepare_adl_data(data, target_col=target_col)

        if forecast_quarter not in full_adl_model.index:
            print(f"Warning: Forecast quarter {forecast_quarter} not in data index. Skipping this step.")
            continue

        x_forecast = full_adl_model.loc[[forecast_quarter], ['GDP_growth_lag1', 'GDP_growth_lag2', "BAA - AAA", "UNRATE", "HOUST"]]
        x_forecast = x_forecast.replace([np.inf, -np.inf], np.nan)  

        if x_forecast.empty or x_forecast.isna().any().any():
            print(f"Warning: Missing predictor data for forecast quarter {forecast_quarter}. Skipping this step.")
            continue

        x_forecast = sm.add_constant(x_forecast, has_constant='add')  # add constant for prediction
        x_forecast = x_forecast.reindex(columns = adl_model.model.exog_names)
        y_pred = adl_model.predict(x_forecast).iloc[0]
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
        print("\nADL Benchmark Results:")
        print(results_df)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Directional Accuracy (Success Ratio): {directional_acc:.3f}")
    else:
        print("\nNo valid forecasts were made in the ADL benchmark evaluation.")
    return results_df


        