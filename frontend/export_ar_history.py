from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.ar_benchmark import fit_ar_benchmark


def build_historical_ar_csv(
    gdp_series,
    output_path=None,
    max_lag=8,
    min_train_size=20,
):
    if output_path is None:
        output_path = ROOT / "data" / "historical_gdp_ar_predictions.csv"

    rows = []

    for i in range(min_train_size, len(gdp_series)):
        train_series = gdp_series.iloc[:i].dropna()
        forecast_quarter = gdp_series.index[i]
        actual = gdp_series.iloc[i]

        model, best_p = fit_ar_benchmark(train_series, max_lag=max_lag)
        predicted = model.forecast(steps=1).iloc[0]

        rows.append({
            "Year and Quarter": str(forecast_quarter).replace("Q", " Q"),
            "Actual GDP growth": actual,
            "AR benchmark predicted GDP growth": predicted,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df