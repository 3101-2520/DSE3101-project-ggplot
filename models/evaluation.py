from config import *
from src.data_preprocessing import aggregate_to_quarterly
from models.bridge_model import fit_bridge_model
from models.ar_indicator import fit_ar_models, fill_ragged_edge

def run_rolling_nowcast(data, md_trans, selected, test_size=8, max_lag=12, target_col='GDP_growth', verbose=VERBOSE):
    """
    Rolling window nowcast evalusation.
    Keep selected variables fixed based on training data.
    Refits bridge model each rolling step.
    Refits AR models for each selected indicator at each step, using all available data up to that quarter.
    Returns a DataFrame with actual vs nowcasted GDP growth for the test period.
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

        # Fit bridge model on current training data
        bridge_model, bridge_coefs = fit_bridge_model(train_data, selected, target_col=target_col)

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
            print(f"Warning: Forecast quarter {forecast_quarter} not in filled quarterly data. Skipping this step.")
            continue
        x_forecast = monthly_q_filled.loc[[forecast_quarter], selected].copy()
        gdp_lag1 = data[target_col].shift(1)
        gdp_lag2 = data[target_col].shift(2)
        x_forecast['GDP_growth_lag1'] = gdp_lag1.loc[forecast_quarter]
        x_forecast['GDP_growth_lag2'] = gdp_lag2.loc[forecast_quarter]
        x_forecast['covid_dummy'] = data.loc[forecast_quarter, 'covid_dummy']
        x_forecast = x_forecast.replace([np.inf, -np.inf], np.nan)

        if x_forecast.empty:
            print(f"Warning: No predictor data available for forecast quarter {forecast_quarter}. Skipping this step.")
            continue

        x_forecast = sm.add_constant(x_forecast, has_constant='add')  # add constant for prediction
        x_forecast = x_forecast.reindex(columns = bridge_model.model.exog_names)

        if verbose:
            print("\nPrediction row shape:", x_forecast.shape)
            print("Prediction row columns:", list(x_forecast.columns))
            print("Model expects columns:", list(bridge_model.model.exog_names))

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

        print("\nRolling nowcast evaluation results:")
        print(results_df)
        print(f"\nRMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("No forecasts were generated. Please check the data and selected variables.")
    return results_df