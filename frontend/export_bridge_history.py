from pathlib import Path
import sys
import pandas as pd
import statsmodels.api as sm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.bridge_model import fit_bridge_model


def build_historical_bridge_csv(
    data,
    selected,
    output_path=None,
    target_col="GDP_growth",
    min_train_size=20,
):
    if output_path is None:
        output_path = ROOT_DIR / "data" / "historical_gdp_bridge_predictions.csv"

    rows = []

    # rows with known actual GDP: used for historical backtesting
    hist_data = data.loc[data[target_col].notna()].copy()
    hist_data["GDP_growth_lag1"] = hist_data[target_col].shift(1)
    hist_data["GDP_growth_lag2"] = hist_data[target_col].shift(2)

    all_predictors = selected + ["GDP_growth_lag1", "GDP_growth_lag2"]
    if "covid_dummy" in hist_data.columns:
        all_predictors.append("covid_dummy")

    # 1) historical predictions where actual GDP is known
    for i in range(min_train_size, len(hist_data)):
        forecast_quarter = hist_data.index[i]
        actual = hist_data.loc[forecast_quarter, target_col]

        try:
            train_data = hist_data.iloc[:i].copy()

            bridge_model, bridge_coefs = fit_bridge_model(
                train_data,
                selected,
                target_col=target_col
            )

            x_forecast = hist_data.loc[[forecast_quarter], all_predictors].copy()
            x_forecast = x_forecast.replace([float("inf"), float("-inf")], pd.NA)

            if x_forecast.isna().any().any():
                print(f"Skipping {forecast_quarter}: missing values in bridge forecast row")
                print(x_forecast.T[x_forecast.isna().iloc[0]])
                continue

            x_forecast = sm.add_constant(x_forecast, has_constant="add")
            x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)
            predicted = bridge_model.predict(x_forecast).iloc[0]

            rows.append({
                "Year and Quarter": str(forecast_quarter).replace("Q", " Q"),
                "Actual GDP growth": actual,
                "Bridge predicted GDP growth": predicted,
            })

        except Exception as e:
            print(f"Failed Bridge at {forecast_quarter}: {e}")
            continue

    # 2) add one extra live/current-quarter forecast as the NEXT quarter after history
    try:
        full_data = data.copy()
        full_data["GDP_growth_lag1"] = full_data[target_col].shift(1)
        full_data["GDP_growth_lag2"] = full_data[target_col].shift(2)

        if "covid_dummy" in full_data.columns and "covid_dummy" not in all_predictors:
            all_predictors.append("covid_dummy")

        if len(hist_data) > 0:
            next_quarter = hist_data.index[-1] + 1

            # only proceed if that next quarter actually exists in full_data
            if next_quarter in full_data.index:
                train_data = hist_data.copy()

                bridge_model, bridge_coefs = fit_bridge_model(
                    train_data,
                    selected,
                    target_col=target_col
                )

                x_forecast = full_data.loc[[next_quarter], all_predictors].copy()
                x_forecast = x_forecast.replace([float("inf"), float("-inf")], pd.NA)

                if x_forecast.isna().any().any():
                    print(f"Skipping live quarter {next_quarter}: missing values in bridge forecast row")
                    print(x_forecast.T[x_forecast.isna().iloc[0]])
                else:
                    x_forecast = sm.add_constant(x_forecast, has_constant="add")
                    x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)
                    predicted = bridge_model.predict(x_forecast).iloc[0]

                    rows.append({
                        "Year and Quarter": str(next_quarter).replace("Q", " Q"),
                        "Actual GDP growth": pd.NA,
                        "Bridge predicted GDP growth": predicted,
                    })

    except Exception as e:
        print(f"Failed live Bridge forecast: {e}")

    df = pd.DataFrame(rows, columns=[
        "Year and Quarter",
        "Actual GDP growth",
        "Bridge predicted GDP growth",
    ])
    df.to_csv(output_path, index=False)
    return df