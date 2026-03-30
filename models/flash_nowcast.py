from config import *
import statsmodels.api as sm

from src.data_preprocessing import aggregate_to_quarterly
from models.bridge_model import fit_bridge_model
from models.ar_indicator import fit_ar_models, fill_ragged_edge


def _get_quarter_months(quarter):
    """
    Return the 3 month-start dates belonging to a quarter Period.
    Example: 2020Q2 -> 2020-04-01, 2020-05-01, 2020-06-01
    """
    start_month = quarter.start_time.normalize()
    return pd.date_range(start=start_month, periods=3, freq="MS")


def _make_flash_monthly_panel(md_trans, forecast_quarter, n_months_available, selected):
    """
    Build a pseudo real-time monthly panel for a given forecast quarter.

    n_months_available:
        1 -> only first month of quarter observed
        2 -> first two months observed
        3 -> all three months observed

    We keep data up to the quarter end, but hide unavailable months in the
    forecast quarter by setting them to NaN. This allows AR filling to create
    flash estimates for the missing months.
    """
    if n_months_available not in [1, 2, 3]:
        raise ValueError("n_months_available must be 1, 2, or 3.")

    quarter_months = _get_quarter_months(forecast_quarter)
    quarter_end = quarter_months[-1]

    # Keep monthly data only up to the end of the forecast quarter
    md_panel = md_trans.loc[:quarter_end, selected].copy()

    # Hide future months within the forecast quarter
    unavailable_months = quarter_months[n_months_available:]
    if len(unavailable_months) > 0:
        md_panel.loc[unavailable_months, selected] = np.nan

    return md_panel


def _build_flash_predictor_row(monthly_q_filled, data, forecast_quarter, selected, target_col):
    """
    Build one quarterly predictor row for the bridge equation.
    """
    if forecast_quarter not in monthly_q_filled.index:
        return None

    x_forecast = monthly_q_filled.loc[[forecast_quarter], selected].copy()
    x_forecast["GDP_growth_lag1"] = data[target_col].shift(1).loc[forecast_quarter]
    x_forecast["GDP_growth_lag2"] = data[target_col].shift(2).loc[forecast_quarter]

    if "covid_dummy" in data.columns:
        x_forecast["covid_dummy"] = data.loc[forecast_quarter, "covid_dummy"]

    x_forecast = x_forecast.replace([np.inf, -np.inf], np.nan)

    if x_forecast.isnull().any().any():
        return None

    return x_forecast


def run_expanding_flash_nowcast(
    data,
    md_trans,
    selected,
    test_size=62,
    max_lag=12,
    target_col="GDP_growth",
    flashes=(1, 2, 3),
    verbose=VERBOSE
):
    """
    Expanding flash nowcast evaluation.

    For each forecast quarter in the last `test_size` quarters:
      - fit the bridge equation on the expanding training window
      - fit monthly AR models on monthly data available up to that quarter
      - generate flash nowcasts using 1, 2, or 3 months of information
        within the target quarter

    Returns a DataFrame with columns:
      quarter, flash, actual, predicted, error
    """
    results = []
    total_obs = len(data)
    train_size = total_obs - test_size

    for i in range(test_size):
        train_end_index = train_size - 1 + i
        forecast_index = train_size + i

        train_data = data.iloc[:train_end_index + 1].copy()
        forecast_quarter = data.index[forecast_index]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Flash nowcasting {forecast_quarter}")
            print(f"Training window: {train_data.index.min()} to {train_data.index.max()}")

        # Add GDP lags to training data
        train_data["GDP_growth_lag1"] = train_data[target_col].shift(1)
        train_data["GDP_growth_lag2"] = train_data[target_col].shift(2)

        all_predictors = selected + ["GDP_growth_lag1", "GDP_growth_lag2"]
        if "covid_dummy" in data.columns:
            all_predictors.append("covid_dummy")

        bridge_model, bridge_coefs = fit_bridge_model(
            train_data,
            all_predictors,
            target_col=target_col
        )

        # Fit AR models once per forecast quarter
        for flash in flashes:
            if verbose:
                print(f"  Flash {flash}: using {flash} month(s) of data")

            quarter_months = _get_quarter_months(forecast_quarter)
            flash_cutoff = quarter_months[flash - 1]

            # 1. Fit AR models only on data available up to this flash cutoff
            md_observed = md_trans.loc[:flash_cutoff, selected].copy()
            ar_models = fit_ar_models(md_observed, selected, max_lag=max_lag)

            # 2. Build flash panel with future months in the forecast quarter hidden
            md_flash = _make_flash_monthly_panel(
                md_trans=md_trans,
                forecast_quarter=forecast_quarter,
                n_months_available=flash,
                selected=selected
            )

            # 3. Fill the hidden months using the AR models fitted above
            md_filled = fill_ragged_edge(md_flash, ar_models, selected)

            # 4. Aggregate to quarterly
            monthly_q_filled = aggregate_to_quarterly(md_filled)

            # 5. Build predictor row and predict
            x_forecast = _build_flash_predictor_row(
                monthly_q_filled=monthly_q_filled,
                data=data,
                forecast_quarter=forecast_quarter,
                selected=selected,
                target_col=target_col
            )

            if x_forecast is None or x_forecast.empty:
                print(f"Warning: Could not build predictor row for {forecast_quarter}, flash {flash}.")
                continue

            x_forecast = sm.add_constant(x_forecast, has_constant="add")
            x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)

            if x_forecast.isnull().any().any():
                print(f"Warning: Missing values remain in x_forecast for {forecast_quarter}, flash {flash}.")
                continue

            y_pred = bridge_model.predict(x_forecast).iloc[0]
            y_actual = data.loc[forecast_quarter, target_col]

            results.append({
                "quarter": forecast_quarter,
                "flash": flash,
                "actual": y_actual,
                "predicted": y_pred,
                "error": y_actual - y_pred
            })


    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No flash nowcasts were generated.")
        return results_df

    print("\nFlash nowcast results: (first 5 rows)")
    print(results_df.head(5))
    print(f"\nFlash nowcast results: (last 5 rows)")
    print(results_df.tail(5))

    print("\nOverall flash nowcast metrics:")
    overall_rmse = np.sqrt((results_df["error"] ** 2).mean())
    overall_mae = np.abs(results_df["error"]).mean()
    overall_da = np.mean(np.sign(results_df["actual"]) == np.sign(results_df["predicted"]))
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"Directional Accuracy (Success Ratio): {overall_da:.3f}")

    # --- Per‑flash metrics ---
    print("\nMetrics by flash:")
    summary = (
        results_df.groupby("flash", group_keys = False)
        .apply(lambda g: pd.Series({
            "RMSE": np.sqrt(np.mean(g["error"] ** 2)),
            "MAE": np.mean(np.abs(g["error"])),
            "DirAcc": np.mean(np.sign(g["actual"]) == np.sign(g["predicted"]))
        }))
        .reset_index()
    )
    print(summary)

    return results_df