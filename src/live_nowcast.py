import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.ar_indicator import fit_ar_models, fill_ragged_edge
from models.bridge_model import fit_bridge_model
from models.flash_nowcast import _make_flash_monthly_panel, _build_flash_predictor_row
from models.ar_benchmark import fit_ar_benchmark
from models.adl_benchmark import fit_adl_benchmark, prepare_adl_data
from src.data_preprocessing import load_and_transform_md, aggregate_to_quarterly, load_and_transform_qd, merge_data, transform_series
from config import *

if __name__ == "__main__":

    # Paths

    ROOT = Path(__file__).resolve().parents[1]
    monthly_path = ROOT / "data/live_api_monthly.csv"
    gdp_path = ROOT / "data/live_api_quarterly_gdp.csv"
    md_reference_path = ROOT / "data/2026-02-MD.csv"
    output_path = ROOT / "data/live_nowcast_results.csv"

    # Load API data
    print("\nLoading API datasets...")
    monthly_raw = pd.read_csv(monthly_path, parse_dates=True, index_col=0)
    gdp_raw = pd.read_csv(gdp_path, parse_dates=True, index_col=0).squeeze()
    print(f"Monthly data shape: {monthly_raw.shape}")
    print(f"Quarterly GDP shape: {gdp_raw.shape}")

    # Transform monthly indicators

    md_meta = pd.read_csv(md_reference_path, nrows=1)
    tcodes = md_meta.iloc[0].to_dict()

    transformed_list = []

    for col in monthly_raw.columns:
        if col not in tcodes:
            continue
        code = tcodes[col]
        if pd.isna(code):
            continue
        try:
            s = transform_series(monthly_raw[col].astype(float), int(code))
            s.name = col
            transformed_list.append(s)
        except:
            continue
    
    MD_trans = pd.concat(transformed_list, axis=1).sort_index()
    print(f"Transformed monthly data shape: {MD_trans.shape}")

    # Drop variables
    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')
    print(f"Dropped {len(vars_to_drop)} variables. New shape: {MD_trans.shape}")

    # Prepare GDP growth

    gdp_raw.index = pd.to_datetime(gdp_raw.index)
    gdp_raw.index = gdp_raw.index.to_period('Q')
    GDP_growth = (np.log(gdp_raw).diff().dropna()*400).rename("GDP_growth")
    print(f"GDP growth observations: {len(GDP_growth)}")
    

    # Selected bridge indicators

    selected = ['DPCERA3M086SBEA', 'UEMP15T26', 'DMANEMP', 'IPDMAT', 'W875RX1', 'UNRATE']
    print(f"Selected bridge indicators: {', '.join(selected)}")

    print("\nChecking selected variables in MD_trans:")
    for var in selected:
        if var in MD_trans.columns:
            print(f"{var} in MD_trans")
        else:
            print(f"Warning: {var} NOT found in MD_trans")

    # Aggregate monthly to quarterly
    monthly_q = aggregate_to_quarterly(MD_trans)
    if not isinstance(monthly_q.index, pd.PeriodIndex):
        monthly_q.index = monthly_q.index.to_period('Q')

    # Keep only required predictors
    required_cols = list(set(selected + ["BAA", "AAA", "HOUST", "UNRATE"]))
    missing_cols = [col for col in required_cols if col not in monthly_q.columns]
    if missing_cols:
        print(f"Warning: The following required columns are missing from the aggregated monthly data: {', '.join(missing_cols)}")

    available_cols = [col for col in required_cols if col in monthly_q.columns]
    monthly_q = monthly_q[available_cols].copy()

    print("\nQuarterly predictors kept for live models:")
    print(available_cols)

    # Live merge, keep target quarter rows even if GDP is missing
    data = monthly_q.join(GDP_growth, how='left')
    print(f"\nMerged dataset shape: {data.shape}")

    # Add covid dummy
    data["covid_dummy"] = 0
    data.loc[(data.index >= pd.Period('2020Q1', freq='Q')) & 
            (data.index <= pd.Period('2020Q4', freq='Q')), 'covid_dummy'] = 1

      # Determine Nowcast Quarter automatically based on latest available month in data

    latest_month = MD_trans.index.max()
    latest_month_q = latest_month.to_period('Q')
    latest_gdp_quarter = GDP_growth.index.max()
    print(f"Latest gdp quarter available: {latest_gdp_quarter}")
    print(f"Latest monthly observation: {latest_month.date()}, corresponding to quarter {latest_month_q}")

    Q1 = latest_gdp_quarter + 1
    Q2 = latest_gdp_quarter + 2

    target_quarter = [Q1]
    if latest_month_q >= Q2:
        target_quarter.append(Q2)
    print(f"Nowcasting quarter(s): {', '.join(str(q) for q in target_quarter)}")

    # Ensure future months for target quarters exist
    for target in target_quarter:
        quarter_months = pd.date_range(start=target.start_time, periods = 3, freq='MS')

        for month in quarter_months:
            if month not in MD_trans.index:
                MD_trans.loc[month] = np.nan
        
        MD_trans = MD_trans.sort_index()

    # Fit bridge regression on quarrters where GDP is known

    print("\nFitting bridge model...")
    train_data = data.loc[data["GDP_growth"].notna()].copy()
    train_data["GDP_growth_lag1"] = train_data["GDP_growth"].shift(1)
    train_data["GDP_growth_lag2"] = train_data["GDP_growth"].shift(2)
    predictors = selected + ["GDP_growth_lag1", "GDP_growth_lag2", "covid_dummy"]
    bridge_model, _ = fit_bridge_model(train_data, predictors)

    # Fit benchmark models
    adl_model = fit_adl_benchmark(train_data)
    ar_model, best_p = fit_ar_benchmark(data["GDP_growth"].dropna())

    # Run nowcast

    results = []

    bridge_target_series = data["GDP_growth"].copy()
    adl_target_series = data["GDP_growth"].copy()

    for target in target_quarter:
        print(f"\nNowcasting {target}...")
        quarter_months = pd.date_range(start=target.start_time, periods = 3, freq='MS')
        
        MD_trans = MD_trans.sort_index()
        
        # Determine flash stages available

        flashes = []

        for i, month in enumerate(quarter_months, start=1):
            if month <= latest_month:
                flashes.append(i)
        print(f"  Available flash stages for {target}: {flashes}")

        bridge_preds = {
            "bridge_flash1": np.nan,
            "bridge_flash2": np.nan,
            "bridge_flash3": np.nan
        }

        if target not in data.index:
            data.loc[target] = np.nan
            data = data.sort_index()
        
        data.loc[target, "covid_dummy"] = 0

        bridge_data_for_forecast = data.copy()
        bridge_data_for_forecast["GDP_growth"] = bridge_target_series


        # Bridge flash predictions
        for flash in flashes:
            flash_cutoff = quarter_months[flash - 1]

            # fit ar models on data available up to flash cutoff

            md_observed = MD_trans.loc[:flash_cutoff, selected].copy()
            ar_models = fit_ar_models(md_observed, selected)

            md_flash = _make_flash_monthly_panel(MD_trans, target, flash, selected)
            md_filled = fill_ragged_edge(md_flash, ar_models, selected)
            monthly_q_filled = aggregate_to_quarterly(md_filled)
            if not isinstance(monthly_q_filled.index, pd.PeriodIndex):
                monthly_q_filled.index = monthly_q_filled.index.to_period('Q')

            x_forecast = _build_flash_predictor_row(
                monthly_q_filled=monthly_q_filled,
                data=bridge_data_for_forecast,
                forecast_quarter=target,
                selected=selected,
                target_col='GDP_growth'
            )

            if x_forecast is None or x_forecast.empty:
                print(f"Warning: No predictor data available for {target} at flash {flash}. Skipping this flash prediction.")
                continue

            x_forecast = sm.add_constant(x_forecast, has_constant='add')
            x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)

            if x_forecast.isna().any().any():
                print(f"Warning: Missing predictor data for {target} at flash {flash}. Skipping this flash prediction.")
                continue

            prediction_results = bridge_model.get_prediction(x_forecast)
            pred = prediction_results.predicted_mean[0]
            se = prediction_results.se_obs[0] # standard error

            bridge_preds[f"bridge_flash{flash}"] = pred
            bridge_preds[f"bridge_flash{flash}_se"] = se # Save it to the CSV
            print(f"Bridge flash {flash} prediction for {target}: {pred:.4f}")

        bridge_recursive_pred = np.nan
        for candidate in ["bridge_flash3", "bridge_flash2", "bridge_flash1"]:
            if not pd.isna(bridge_preds[candidate]):
                bridge_recursive_pred = bridge_preds[candidate]
                break
        
        if not pd.isna(bridge_recursive_pred):
            bridge_target_series.loc[target] = bridge_recursive_pred

    # ADL benchmark prediction
        adl_pred = np.nan
        try:
            adl_data_for_forecast = data.copy()
            adl_data_for_forecast["GDP_growth"] = adl_target_series
            adl_full = prepare_adl_data(adl_data_for_forecast, target_col="GDP_growth")
            adl_feature_cols = ["GDP_growth_lag1", "GDP_growth_lag2", "BAA - AAA_lag1", "UNRATE_lag1", "HOUST_lag1"]
            if target in adl_full.index:
                x_adl_forecast = adl_full.loc[[target], adl_feature_cols].copy()
                x_adl_forecast = x_adl_forecast.replace([np.inf, -np.inf], np.nan)
                if not x_adl_forecast.isna().any().any() and not x_adl_forecast.empty:
                    x_adl_forecast = sm.add_constant(x_adl_forecast, has_constant='add')
                    x_adl_forecast = x_adl_forecast.reindex(columns=adl_model.model.exog_names)
                    adl_pred = adl_model.predict(x_adl_forecast).iloc[0]
                    adl_target_series.loc[target] = adl_pred
                    print(f"ADL benchmark prediction for {target}: {adl_pred:.4f}")
                else:
                    print(f"Warning: Missing predictor data for ADL benchmark at {target}. Skipping ADL prediction.")
        
        except Exception as e:
            print(f"Error during ADL benchmark prediction for {target}: {e}")
        
        # AR benchmark prediction
        ar_pred = np.nan

        try:
            steps_ahead = target.ordinal - latest_gdp_quarter.ordinal
            if steps_ahead >= 1:
                ar_forecasts = ar_model.forecast(steps=steps_ahead)
                ar_pred = ar_forecasts.iloc[-1]
                print(f"AR benchmark prediction for {target}: {ar_pred:.4f}")
            else:
                print(f"Warning: Target quarter {target} is not ahead of latest GDP quarter {latest_gdp_quarter}. Skipping AR benchmark prediction.")
        
        except Exception as e:
            print(f"Error during AR benchmark prediction for {target}: {e}")

        row = {
            "quarter": target,
            "bridge_flash1": bridge_preds.get("bridge_flash1"),
            "bridge_flash1_se": bridge_preds.get("bridge_flash1_se"), 
            "bridge_flash2": bridge_preds.get("bridge_flash2"),
            "bridge_flash2_se": bridge_preds.get("bridge_flash2_se"), 
            "bridge_flash3": bridge_preds.get("bridge_flash3"),
            "bridge_flash3_se": bridge_preds.get("bridge_flash3_se"), 
            "adl_benchmark": adl_pred,
            "ar_benchmark": ar_pred
            }

        results.append(row)

        print("\nFinal prediction row:")
        print(pd.DataFrame([row]))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("quarter").reset_index(drop=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nNowcast results saved to {output_path}")
    print("\nNowcast results:")
    print(results_df)