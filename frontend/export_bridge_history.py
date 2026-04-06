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

    hist_data = data.loc[data[target_col].notna()].copy()
    hist_data["GDP_growth_lag1"] = hist_data[target_col].shift(1)
    hist_data["GDP_growth_lag2"] = hist_data[target_col].shift(2)

    all_predictors = selected + ["GDP_growth_lag1", "GDP_growth_lag2"]
    if "covid_dummy" in hist_data.columns:
        all_predictors.append("covid_dummy")

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

    df = pd.DataFrame(rows, columns=[
        "Year and Quarter",
        "Actual GDP growth",
        "Bridge predicted GDP growth",
    ])
    df.to_csv(output_path, index=False)
    return df