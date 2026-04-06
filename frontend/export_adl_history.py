from pathlib import Path
import pandas as pd
import sys
import statsmodels.api as sm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.adl_benchmark import fit_adl_benchmark, prepare_adl_data


def build_historical_adl_csv(
    data,
    output_path=None,
    target_col="GDP_growth",
    min_train_size=20,
):
    if output_path is None:
        output_path = ROOT_DIR / "data" / "historical_gdp_adl_predictions.csv"

    rows = []

    # only use quarters where GDP is actually known for historical evaluation
    hist_data = data.loc[data[target_col].notna()].copy()

    feature_cols = [
        "GDP_growth_lag1",
        "GDP_growth_lag2",
        "BAA - AAA_lag1",
        "UNRATE_lag1",
        "HOUST_lag1",
    ]

    # 1) historical backtest
    for i in range(min_train_size, len(hist_data)):
        forecast_quarter = hist_data.index[i]
        actual = hist_data.loc[forecast_quarter, target_col]

        try:
            train_data = hist_data.iloc[:i].copy()

            adl_model = fit_adl_benchmark(
                train_data,
                target_col=target_col,
                verbose=False
            )

            full_adl_data = prepare_adl_data(hist_data.copy(), target_col=target_col)

            if forecast_quarter not in full_adl_data.index:
                print(f"Skipping {forecast_quarter}: not in ADL feature table")
                continue

            missing_feature_cols = [c for c in feature_cols if c not in full_adl_data.columns]
            if missing_feature_cols:
                print(f"Skipping {forecast_quarter}: missing ADL columns {missing_feature_cols}")
                continue

            x_forecast = full_adl_data.loc[[forecast_quarter], feature_cols].copy()
            x_forecast = x_forecast.replace([float("inf"), float("-inf")], pd.NA)

            if x_forecast.isna().any().any():
                print(f"Skipping {forecast_quarter}: missing values in ADL forecast row")
                print(x_forecast.T[x_forecast.isna().iloc[0]])
                continue

            x_forecast = sm.add_constant(x_forecast, has_constant="add")
            x_forecast = x_forecast.reindex(columns=adl_model.model.exog_names)

            predicted = adl_model.predict(x_forecast).iloc[0]

            rows.append({
                "Year and Quarter": str(forecast_quarter).replace("Q", " Q"),
                "Actual GDP growth": actual,
                "ADL benchmark predicted GDP growth": predicted,
            })

        except Exception as e:
            print(f"Failed ADL at {forecast_quarter}: {e}")
            continue

    # 2) one extra live forecast for the next quarter
    try:
        if len(hist_data) > 0:
            next_quarter = hist_data.index[-1] + 1

            if next_quarter in data.index:
                train_data = hist_data.copy()

                adl_model = fit_adl_benchmark(
                    train_data,
                    target_col=target_col,
                    verbose=False
                )

                full_adl_data = prepare_adl_data(data.copy(), target_col=target_col)

                if next_quarter not in full_adl_data.index:
                    print(f"Skipping live quarter {next_quarter}: not in ADL feature table")
                else:
                    missing_feature_cols = [c for c in feature_cols if c not in full_adl_data.columns]
                    if missing_feature_cols:
                        print(f"Skipping live quarter {next_quarter}: missing ADL columns {missing_feature_cols}")
                    else:
                        x_forecast = full_adl_data.loc[[next_quarter], feature_cols].copy()
                        x_forecast = x_forecast.replace([float('inf'), float('-inf')], pd.NA)

                        if x_forecast.isna().any().any():
                            print(f"Skipping live quarter {next_quarter}: missing values in ADL forecast row")
                            print(x_forecast.T[x_forecast.isna().iloc[0]])
                        else:
                            x_forecast = sm.add_constant(x_forecast, has_constant="add")
                            x_forecast = x_forecast.reindex(columns=adl_model.model.exog_names)

                            predicted = adl_model.predict(x_forecast).iloc[0]

                            rows.append({
                                "Year and Quarter": str(next_quarter).replace("Q", " Q"),
                                "Actual GDP growth": pd.NA,
                                "ADL benchmark predicted GDP growth": predicted,
                            })

    except Exception as e:
        print(f"Failed live ADL forecast: {e}")

    df = pd.DataFrame(rows, columns=[
        "Year and Quarter",
        "Actual GDP growth",
        "ADL benchmark predicted GDP growth",
    ])
    df.to_csv(output_path, index=False)
    return df