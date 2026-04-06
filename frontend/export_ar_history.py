from pathlib import Path
import pandas as pd
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.ar_benchmark import fit_ar_benchmark


def build_historical_ar_csv(
    gdp_series,
    output_path=None,
    max_lag=8,
    min_train_size=20,
):
    if output_path is None:
        output_path = ROOT_DIR / "data" / "historical_gdp_ar_predictions.csv"

    rows = []

    hist_series = gdp_series.dropna().copy()

    # historical backtest
    for i in range(min_train_size, len(hist_series)):
        train_series = hist_series.iloc[:i].copy()
        forecast_quarter = hist_series.index[i]
        actual = hist_series.iloc[i]

        try:
            model, best_p = fit_ar_benchmark(train_series, max_lag=max_lag)
            predicted = model.forecast(steps=1).iloc[0]

            rows.append({
                "Year and Quarter": str(forecast_quarter).replace("Q", " Q"),
                "Actual GDP growth": actual,
                "AR benchmark predicted GDP growth": predicted,
            })

        except Exception as e:
            print(f"Failed AR at {forecast_quarter}: {e}")
            continue

    # add one extra next-quarter forecast even if actual GDP is unavailable
    try:
        if len(hist_series) > 0:
            next_quarter = hist_series.index[-1] + 1
            model, best_p = fit_ar_benchmark(hist_series.copy(), max_lag=max_lag)
            predicted = model.forecast(steps=1).iloc[0]

            rows.append({
                "Year and Quarter": str(next_quarter).replace("Q", " Q"),
                "Actual GDP growth": pd.NA,
                "AR benchmark predicted GDP growth": predicted,
            })

    except Exception as e:
        print(f"Failed live AR forecast: {e}")

    df = pd.DataFrame(rows, columns=[
        "Year and Quarter",
        "Actual GDP growth",
        "AR benchmark predicted GDP growth",
    ])
    df.to_csv(output_path, index=False)
    return df