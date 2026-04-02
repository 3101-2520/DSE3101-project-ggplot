from pathlib import Path
import pandas as pd
import sys
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.adl_benchmark import fit_adl_benchmark, prepare_adl_data


def build_historical_adl_csv(
    data,
    output_path=None,
    target_col="GDP_growth",
    min_train_size=20,
):
    if output_path is None:
        output_path = ROOT / "data" / "historical_gdp_adl_predictions.csv"

    rows = []

    for i in range(min_train_size, len(data)):
        train_data = data.iloc[:i].copy()
        forecast_quarter = data.index[i]
        actual = data.loc[forecast_quarter, target_col]

        adl_model = fit_adl_benchmark(
            train_data,
            target_col=target_col,
            verbose=False
        )

        full_adl_data = prepare_adl_data(data, target_col=target_col)

        x_forecast = full_adl_data.loc[
            [forecast_quarter],
            ["GDP_growth_lag1", "GDP_growth_lag2", "BAA_AAA_lag2", "UNRATE_lag2", "HOUST_lag2"]
        ].copy()

        if x_forecast.isna().any().any():
            continue

        x_forecast = sm.add_constant(x_forecast, has_constant="add")
        x_forecast = x_forecast.reindex(columns=adl_model.model.exog_names)

        predicted = adl_model.predict(x_forecast).iloc[0]

        rows.append({
            "Year and Quarter": str(forecast_quarter).replace("Q", " Q"),
            "Actual GDP growth": actual,
            "ADL benchmark predicted GDP growth": predicted,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df