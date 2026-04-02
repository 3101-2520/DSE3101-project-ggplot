from pathlib import Path
import sys
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.bridge_model import fit_bridge_model


def build_historical_bridge_csv(
    data,
    selected,
    output_path=None,
    target_col="GDP_growth",
    min_train_size=20,
):
    if output_path is None:
        output_path = ROOT / "data" / "historical_gdp_bridge_predictions.csv"

    rows = []

    full_data = data.copy()
    full_data["GDP_growth_lag1"] = full_data[target_col].shift(1)
    full_data["GDP_growth_lag2"] = full_data[target_col].shift(2)

    all_predictors = selected + ["GDP_growth_lag1", "GDP_growth_lag2"]
    if "covid_dummy" in full_data.columns:
        all_predictors.append("covid_dummy")

    for i in range(min_train_size, len(full_data)):
        forecast_quarter = full_data.index[i]
        actual = full_data.loc[forecast_quarter, target_col]

        try:
            train_data = full_data.iloc[:i].copy()

            bridge_model, bridge_coefs = fit_bridge_model(
                train_data,
                selected,
                target_col=target_col
            )

            x_forecast = full_data.loc[[forecast_quarter], all_predictors].copy()
            x_forecast = x_forecast.replace([float("inf"), float("-inf")], pd.NA)

            if x_forecast.isna().any().any():
                continue

            x_forecast = sm.add_constant(x_forecast, has_constant="add")
            x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)
            predicted = bridge_model.predict(x_forecast).iloc[0]

            rows.append({
                "Year and Quarter": str(forecast_quarter).replace("Q", " Q"),
                "Actual GDP growth": actual,
                "Bridge predicted GDP growth": predicted,
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df