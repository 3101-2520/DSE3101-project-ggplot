from config import *
from src.data_preprocessing import aggregate_to_quarterly
from models.bridge_model import fit_bridge_model
from models.ar_indicator import fit_ar_models, fill_ragged_edge
from models.random_forest import fit_rf_model, predict_rf_model

<<<<<<< HEAD

# def run_rolling_nowcast(data, md_trans, selected, test_size=8, max_lag=12, target_col='GDP_growth', verbose=VERBOSE):
#     """
#     Rolling window nowcast evalusation.
#     Keep selected variables fixed based on training data.
#     Refits bridge model each rolling step.
#     Refits AR models for each selected indicator at each step, using all available data up to that quarter.
#     Returns a DataFrame with actual vs nowcasted GDP growth for the test period.
#     """
#     results = []
#     total_obs = len(data)
#     train_size = total_obs - test_size

#     for i in range(test_size):
#         train_end_index = train_size - 1 + i
#         forecast_index = train_size + i

#         train_data = data.iloc[i:train_end_index + 1].copy()
#         forecast_quarter = data.index[forecast_index]

#         if verbose:
#              print(f"\nRolling step {i+1}/{test_size}")
#              print(f"Training window: {train_data.index.min()} to {train_data.index.max()}")
#              print(f"Forecast quarter: {forecast_quarter}")

#         # Fit bridge model on current training data
#         bridge_model, bridge_coefs = fit_bridge_model(train_data, selected, target_col=target_col)

#         # Restrict monthly data to training period for AR fitting
#         forecast_end_date = forecast_quarter.end_time
#         md_window = md_trans.loc[:forecast_end_date].copy()

#         # Fit AR models for selected indicators using data up to forecast quarter
#         ar_models = fit_ar_models(md_window, selected, max_lag=max_lag)

#         # Fill ragged edge for monthly indicators up to forecast quarter
#         md_filled = fill_ragged_edge(md_window, ar_models, selected)

#         # Aggregate filled monthly data to quarterly
#         monthly_q_filled = aggregate_to_quarterly(md_filled)

#         # Extract predictor row for forecast quarter
#         if forecast_quarter not in monthly_q_filled.index:
#             print(f"Warning: Forecast quarter {forecast_quarter} not in filled quarterly data. Skipping this step.")
#             continue
#         x_forecast = monthly_q_filled.loc[[forecast_quarter], selected].copy()
#         x_forecast = x_forecast.replace([np.inf, -np.inf], np.nan)

#         if x_forecast.empty:
#             print(f"Warning: No predictor data available for forecast quarter {forecast_quarter}. Skipping this step.")
#             continue

#         x_forecast = sm.add_constant(x_forecast, has_constant='add')  # add constant for prediction
#         x_forecast = x_forecast.reindex(columns = bridge_model.model.exog_names)

#         if verbose:
#             print("\nPrediction row shape:", x_forecast.shape)
#             print("Prediction row columns:", list(x_forecast.columns))
#             print("Model expects columns:", list(bridge_model.model.exog_names))

#         # Nowcast GDP growth for forecast quarter
#         y_pred = bridge_model.predict(x_forecast).iloc[0]
#         y_actual = data.loc[forecast_quarter, target_col]

#         results.append({
#             'quarter': forecast_quarter,
#             'actual': y_actual,
#             'predicted': y_pred
#         })

#     results_df = pd.DataFrame(results)
#     if not results_df.empty:
#         results_df["error"] = results_df["actual"] - results_df["predicted"]
#         rmse = np.sqrt((results_df["error"] ** 2).mean())
#         mae = np.mean(np.abs(results_df["error"]))

#         print("\nRolling nowcast evaluation results:")
#         print(results_df)
#         print(f"\nRMSE: {rmse:.4f}")
#         print(f"MAE: {mae:.4f}")
#     else:
#         print("No forecasts were generated. Please check the data and selected variables.")
#     return results_df





def run_rolling_nowcast(data, md_trans, selected, test_size=8, max_lag=12,
                        target_col='GDP_growth', include_gdp_lags=False,
                        verbose=VERBOSE):
    """
    Rolling window nowcast evaluation.
    Keeps selected variables fixed based on training data.
    Refits bridge model each rolling step.
    Refits AR models for each selected indicator at each step, using all available data up to that quarter.

    If include_gdp_lags=True, the bridge model is augmented with GDP_growth_lag1 and GDP_growth_lag2.

    Returns a DataFrame with actual vs predicted GDP growth for the test period.
=======
def run_rolling_nowcast(data, md_trans, selected, test_size=8, window_size=80,
                        max_lag=12, target_col='GDP_growth', verbose=VERBOSE):
    """
    Rolling window nowcast evaluation with fixed window size.
    Adds two lags of GDP and a COVID dummy as additional predictors.
    For each forecast quarter (the last test_size quarters), trains on the most recent
    window_size quarters of data, refits bridge model and AR models, then nowcasts.
>>>>>>> 8aff10ace0e46f40544ddabd710c50887602fd34
    """
    results = []
    total_obs = len(data)
    # The indices of the last test_size quarters
    forecast_indices = list(range(total_obs - test_size, total_obs))

    for idx in forecast_indices:
        # Training window: window_size quarters before the forecast quarter
        start_idx = idx - window_size
        if start_idx < 0:
            print(f"Warning: Not enough data to form a window of {window_size} quarters before {data.index[idx]}. Skipping.")
            continue

        train_data = data.iloc[start_idx:idx].copy()
        forecast_quarter = data.index[idx]

        if verbose:
<<<<<<< HEAD
            print(f"\nRolling step {i+1}/{test_size}")
            print(f"Training window: {train_data.index.min()} to {train_data.index.max()}")
            print(f"Forecast quarter: {forecast_quarter}")

        # Fit bridge model on current training data
        bridge_model, bridge_coefs = fit_bridge_model(
            train_data,
            selected,
            target_col=target_col,
            include_gdp_lags=include_gdp_lags,
            verbose=verbose
        )
=======
            print(f"\nNowcasting {forecast_quarter}")
            print(f"Training window: {train_data.index.min()} to {train_data.index.max()}")

        # Create lagged GDP and ensure COVID dummy exists
        train_data['GDP_growth_lag1'] = train_data[target_col].shift(1)
        train_data['GDP_growth_lag2'] = train_data[target_col].shift(2)
        # Combine predictors
        all_predictors = selected + ['GDP_growth_lag1', 'GDP_growth_lag2', 'covid_dummy']

        # Fit bridge model on current training data with all predictors
        bridge_model, bridge_coefs = fit_bridge_model(train_data, all_predictors, target_col=target_col)
>>>>>>> 8aff10ace0e46f40544ddabd710c50887602fd34

        # Restrict monthly data to training period for AR fitting
        forecast_end_date = forecast_quarter.end_time
        md_window = md_trans.loc[:forecast_end_date].copy()

        # Fit AR models for selected indicators using data up to forecast quarter
        ar_models = fit_ar_models(md_window, selected, max_lag=max_lag)

        # Fill ragged edge for monthly indicators up to forecast quarter
        md_filled = fill_ragged_edge(md_window, ar_models, selected)

        # Aggregate filled monthly data to quarterly
        monthly_q_filled = aggregate_to_quarterly(md_filled)

        # Extract predictor row for forecast quarter
        if forecast_quarter not in monthly_q_filled.index:
            print(f"Warning: Forecast quarter {forecast_quarter} not in filled quarterly data. Skipping.")
            continue

        x_forecast = monthly_q_filled.loc[[forecast_quarter], selected].copy()
            # Add lags and dummy from the original data (these are known at forecast time)
            x_forecast['GDP_growth_lag1'] = data[target_col].shift(1).loc[forecast_quarter]
            x_forecast['GDP_growth_lag2'] = data[target_col].shift(2).loc[forecast_quarter]
            x_forecast['covid_dummy'] = data.loc[forecast_quarter, 'covid_dummy']

        x_forecast = x_forecast.replace([np.inf, -np.inf], np.nan)

<<<<<<< HEAD
        # Add lagged GDP terms if requested
        if include_gdp_lags:
            data_with_lags = data.copy()
            data_with_lags["GDP_growth_lag1"] = data_with_lags[target_col].shift(1)
            data_with_lags["GDP_growth_lag2"] = data_with_lags[target_col].shift(2)

            if forecast_quarter not in data_with_lags.index:
                print(f"Warning: Forecast quarter {forecast_quarter} not in lagged GDP data. Skipping this step.")
                continue

            x_forecast["GDP_growth_lag1"] = data_with_lags.loc[forecast_quarter, "GDP_growth_lag1"]
            x_forecast["GDP_growth_lag2"] = data_with_lags.loc[forecast_quarter, "GDP_growth_lag2"]

        if x_forecast.empty or x_forecast.isna().any().any():
            print(f"Warning: No valid predictor data available for forecast quarter {forecast_quarter}. Skipping this step.")
            continue

        x_forecast = sm.add_constant(x_forecast, has_constant='add')
        x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)

        if verbose:
            print("\nPrediction row shape:", x_forecast.shape)
            print("Prediction row columns:", list(x_forecast.columns))
            print("Model expects columns:", list(bridge_model.model.exog_names))
=======
        if x_forecast.empty:
            print(f"Warning: No predictor data for {forecast_quarter}. Skipping.")
            continue

        # Reindex to match the model's expected columns (including constant)
        x_forecast = sm.add_constant(x_forecast, has_constant='add')
        x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)
>>>>>>> 8aff10ace0e46f40544ddabd710c50887602fd34

        # Nowcast GDP growth for forecast quarter
        y_pred = bridge_model.predict(x_forecast).iloc[0]
        y_actual = data.loc[forecast_quarter, target_col]

        results.append({
            'quarter': forecast_quarter,
            'actual': y_actual,
            'predicted': y_pred
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["error"] = results_df["actual"] - results_df["predicted"]
        rmse = np.sqrt((results_df["error"] ** 2).mean())
        mae = np.mean(np.abs(results_df["error"]))

        if include_gdp_lags:
            print("\nRolling nowcast evaluation results (Bridge + GDP lags):")
        else:
            print("\nRolling nowcast evaluation results (Bridge):")

        print(results_df)
        print(f"\nRMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("No forecasts were generated. Please check the data and selected variables.")

    return results_df


def run_rf_benchmark(data, selected, test_size=8, target_col='GDP_growth',
                     rf_params=None, verbose=VERBOSE):
    """
    Rolling/recursive evaluation for low-frequency Random Forest benchmark.
    Uses quarterly data directly.
    Keeps selected variables fixed and refits RF each step.
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
            print(f"\nRF rolling step {i+1}/{test_size}")
            print(f"Training window: {train_data.index.min()} to {train_data.index.max()}")
            print(f"Forecast quarter: {forecast_quarter}")

        # Fit RF on current rolling training window
        rf_model = fit_rf_model(
            train_data=train_data,
            feature_cols=selected,
            target_col=target_col,
            rf_params=rf_params
        )

        # Forecast row
        x_forecast = data.loc[[forecast_quarter], selected].copy()
        x_forecast = x_forecast.replace([np.inf, -np.inf], np.nan)

        if x_forecast.empty or x_forecast.isnull().any().any():
            print(f"Warning: Invalid predictor data for forecast quarter {forecast_quarter}. Skipping this step.")
            continue

        # Predict
        y_pred = predict_rf_model(rf_model, x_forecast)
        y_actual = data.loc[forecast_quarter, target_col]

        results.append({
            'quarter': forecast_quarter,
            'actual': y_actual,
            'predicted': y_pred
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["error"] = results_df["actual"] - results_df["predicted"]
        rmse = np.sqrt((results_df["error"] ** 2).mean())
        mae = np.mean(np.abs(results_df["error"]))

        print("\nRF benchmark evaluation results:")
        print(results_df)
        print(f"\nRMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("No RF forecasts were generated. Please check the data and selected variables.")

    return results_df