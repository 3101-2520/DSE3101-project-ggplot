import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys

# Ensure the script can find your src and models folders
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.bridge_model import fit_bridge_model
from models.ar_indicator import fit_ar_models, fill_ragged_edge
from models.flash_nowcast import _make_flash_monthly_panel, _build_flash_predictor_row
from src.data_preprocessing import aggregate_to_quarterly


def overwrite_next_quarter_with_live_bridge(df_evolution, full_data, live_nowcast_path=None, target_col="GDP_growth"):
    """
    For the next quarter after the last actual GDP quarter:
    - overwrite nowcast_month 1,2,3 using live_nowcast_results.csv
    - overwrite nowcast_month 4 with the latest available live flash
    """
    if live_nowcast_path is None:
        live_nowcast_path = ROOT_DIR / "data" / "live_nowcast_results.csv"

    try:
        live_df = pd.read_csv(live_nowcast_path)

        if live_df.empty or "quarter" not in live_df.columns:
            return df_evolution

        hist_quarters = full_data.index[full_data[target_col].notna()]
        if len(hist_quarters) == 0:
            return df_evolution

        target_quarter_period = hist_quarters.max() + 1
        target_quarter = str(target_quarter_period).replace("Q", " Q")
        target_quarter_clean = str(target_quarter_period)

        live_df["quarter_clean"] = live_df["quarter"].astype(str).str.replace(" ", "", regex=False)
        live_match = live_df.loc[live_df["quarter_clean"] == target_quarter_clean]

        if live_match.empty:
            return df_evolution

        live_row = live_match.iloc[0]

        flash1 = live_row.get("bridge_flash1", np.nan)
        flash2 = live_row.get("bridge_flash2", np.nan)
        flash3 = live_row.get("bridge_flash3", np.nan)

        # month 4 = latest available live flash
        month4 = np.nan
        for v in [flash3, flash2, flash1]:
            if pd.notna(v):
                month4 = v
                break

        # remove existing rows for this target quarter, months 1-4
        mask_remove = (
            (df_evolution["target_quarter"] == target_quarter) &
            (df_evolution["nowcast_month"].isin([1, 2, 3, 4]))
        )
        df_evolution = df_evolution.loc[~mask_remove].copy()

        new_rows = []
        flash_map = {
            1: flash1,
            2: flash2,
            3: flash3,
            4: month4,
        }

        for m, val in flash_map.items():
            if pd.notna(val):
                new_rows.append({
                    "target_quarter": target_quarter,
                    "nowcast_month": m,
                    "prediction": val * 100
                })

        if new_rows:
            df_evolution = pd.concat([df_evolution, pd.DataFrame(new_rows)], ignore_index=True)

        df_evolution = df_evolution.sort_values(["target_quarter", "nowcast_month"]).reset_index(drop=True)
        return df_evolution

    except Exception as e:
        print(f"Warning: could not overwrite next quarter with live bridge values: {e}")
        return df_evolution
    


def build_bridge_evolution_csv(data, md_trans, selected_names, output_path=None, target_col='GDP_growth', min_train_size=20):
    if output_path is None:
        output_path = ROOT_DIR / "data" / "bridge_evolution.csv"

    results = []

    full_data = data.copy()
    full_data["GDP_growth_lag1"] = full_data[target_col].shift(1)
    full_data["GDP_growth_lag2"] = full_data[target_col].shift(2)

    if "covid_dummy" not in full_data.columns:
        full_data["covid_dummy"] = 0

    all_predictors = selected_names + ["GDP_growth_lag1", "GDP_growth_lag2", "covid_dummy"]

    if not md_trans.empty:
        max_quarter = full_data.index.max()
        if pd.notna(max_quarter):
            full_idx = pd.date_range(
                start=md_trans.index.min(),
                end=max_quarter.end_time,
                freq="MS"
            )
            md_trans = md_trans.reindex(full_idx)

    hist_quarters = full_data.index[full_data[target_col].notna()]
    last_actual_quarter = hist_quarters.max() if len(hist_quarters) > 0 else None
    next_live_quarter = last_actual_quarter + 1 if last_actual_quarter is not None else None

    latest_month = md_trans.dropna(how="all").index.max() if not md_trans.empty else None

    print("Starting rigorous historical flash simulation... This may take a minute or two.")

    for i in range(min_train_size, len(full_data)):
        target_quarter = full_data.index[i]
        target_q_str = str(target_quarter).replace("Q", " Q")

        if pd.isna(full_data.loc[target_quarter, target_col]) and target_quarter != next_live_quarter:
            print(f"  [-] Skipping {target_q_str}: No Actual GDP data released by FRED yet.")
            continue

        train_data = full_data.iloc[:i].copy().dropna(subset=[target_col])

        try:
            bridge_model, _ = fit_bridge_model(train_data, selected_names, target_col=target_col)
        except Exception as e:
            print(f"  [X] Skipped {target_q_str} entirely: Bridge model training failed ({e})")
            continue

        quarter_months = pd.date_range(start=target_quarter.start_time, periods=3, freq="MS")

        if target_quarter != next_live_quarter:
            flashes_to_run = [1, 2, 3]
        else:
            flashes_to_run = []
            if latest_month is not None:
                for flash, month in enumerate(quarter_months, start=1):
                    if month <= latest_month:
                        flashes_to_run.append(flash)

        # Flash 1-3
        for flash in flashes_to_run:
            try:
                flash_cutoff = quarter_months[flash - 1]
                md_observed = md_trans.loc[:flash_cutoff, selected_names].copy()
                ar_models = fit_ar_models(md_observed, selected_names)

                md_flash = _make_flash_monthly_panel(md_trans, target_quarter, flash, selected_names)
                md_filled = fill_ragged_edge(md_flash, ar_models, selected_names)
                monthly_q_filled = aggregate_to_quarterly(md_filled)

                if not isinstance(monthly_q_filled.index, pd.PeriodIndex):
                    monthly_q_filled.index = monthly_q_filled.index.to_period("Q")

                x_forecast = _build_flash_predictor_row(
                    monthly_q_filled=monthly_q_filled,
                    data=full_data,
                    forecast_quarter=target_quarter,
                    selected=selected_names,
                    target_col=target_col
                )

                if x_forecast is None or x_forecast.empty:
                    print(f"  [!] Skipped {target_q_str} Flash {flash}: Not enough data yet.")
                    continue

                x_forecast["GDP_growth_lag1"] = full_data.loc[target_quarter - 1, target_col]
                x_forecast["GDP_growth_lag2"] = full_data.loc[target_quarter - 2, target_col]
                x_forecast["covid_dummy"] = full_data.loc[target_quarter, "covid_dummy"]

                x_forecast = sm.add_constant(x_forecast, has_constant="add")
                x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)

                if x_forecast.isna().any().any():
                    missing_cols = x_forecast.columns[x_forecast.isna().any()].tolist()
                    print(f"  [!] Skipped {target_q_str} Flash {flash} due to missing data: {missing_cols}")
                    continue

                pred = float(np.squeeze(bridge_model.predict(x_forecast).values))
                results.append({
                    "target_quarter": target_q_str,
                    "nowcast_month": flash,
                    "prediction": pred
                })

            except Exception as e:
                print(f"  [X] CRITICAL ERROR on {target_q_str} Flash {flash}: {e}")

        # Flash 4 / month after third month
        try:
            x_final = full_data.loc[[target_quarter], all_predictors].copy()
            x_final["GDP_growth_lag1"] = full_data.loc[target_quarter - 1, target_col]
            x_final["GDP_growth_lag2"] = full_data.loc[target_quarter - 2, target_col]
            x_final["covid_dummy"] = full_data.loc[target_quarter, "covid_dummy"]

            x_final = sm.add_constant(x_final, has_constant="add")
            x_final = x_final.reindex(columns=bridge_model.model.exog_names)

            if not x_final.isna().any().any():
                final_pred = float(np.squeeze(bridge_model.predict(x_final).values))
                results.append({
                    "target_quarter": target_q_str,
                    "nowcast_month": 4,
                    "prediction": final_pred
                })
            else:
                missing_cols = x_final.columns[x_final.isna().any()].tolist()
                print(f"  [!] Skipped {target_q_str} Flash 4 due to missing data: {missing_cols}")

        except Exception as e:
            print(f"  [X] CRITICAL ERROR on {target_q_str} Flash 4: {e}")

    df_evolution = pd.DataFrame(results)

    # overwrite only 2026Q1 flash 1-3 with live_nowcast_results.csv; keep flash 4 from this script
    df_evolution = pd.DataFrame(results)

    df_evolution = overwrite_next_quarter_with_live_bridge(
        df_evolution,
        full_data,
        target_col=target_col
    )

    if output_path is not None:
        df_evolution.to_csv(output_path, index=False)
        print(f"\n✅ Successfully exported accurate evolution data to {output_path}")

    return df_evolution


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Loading LIVE datasets for Evolution Chart...")

    monthly_path = ROOT_DIR / "data" / "live_api_monthly.csv"
    gdp_path = ROOT_DIR / "data" / "live_api_quarterly_gdp.csv"
    md_reference_path = ROOT_DIR / "data" / "2026-02-MD.csv" 
    output_path = ROOT_DIR / "data" / "bridge_evolution.csv"

    monthly_raw = pd.read_csv(monthly_path, parse_dates=True, index_col=0)
    md_meta = pd.read_csv(md_reference_path, nrows=1)
    tcodes = md_meta.iloc[0].to_dict()

    from src.data_preprocessing import transform_series

    transformed_list = []
    for col in monthly_raw.columns:
        if col not in tcodes or pd.isna(tcodes[col]):
            continue
        try:
            s = transform_series(monthly_raw[col].astype(float), int(tcodes[col]))
            s.name = col
            transformed_list.append(s)
        except Exception:
            continue

    MD_trans = pd.concat(transformed_list, axis=1).sort_index()
    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

    gdp_raw = pd.read_csv(gdp_path, parse_dates=True, index_col=0).squeeze()
    gdp_raw.index = pd.to_datetime(gdp_raw.index).to_period("Q")
    GDP_growth = (np.log(gdp_raw).diff() * 400).rename("GDP_growth")

    monthly_q = aggregate_to_quarterly(MD_trans)
    if not isinstance(monthly_q.index, pd.PeriodIndex):
        monthly_q.index = monthly_q.index.to_period("Q")
        
    data = monthly_q.join(GDP_growth, how="left")

    data["covid_dummy"] = 0
    data.loc[
        (data.index >= pd.Period("2020Q1", freq="Q")) &
        (data.index <= pd.Period("2020Q4", freq="Q")),
        "covid_dummy"
    ] = 1

    selected = ['DPCERA3M086SBEA', 'UEMP15T26', 'DMANEMP', 'IPDMAT', 'W875RX1', 'UNRATE']
    available_selected = [s for s in selected if s in data.columns]

    build_bridge_evolution_csv(
        data=data,
        md_trans=MD_trans,
        selected_names=available_selected,
        output_path=output_path,
        target_col="GDP_growth",
        min_train_size=20
    )

