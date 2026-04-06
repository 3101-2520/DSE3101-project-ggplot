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

def build_bridge_evolution_csv(data, md_trans, selected_names, output_path=None, target_col='GDP_growth', min_train_size=20):
    if output_path is None:
        output_path = ROOT_DIR / "data" / "bridge_evolution.csv"

    results = []
    
    # 1. Prepare full dataset with the correct lags and dummies
    full_data = data.copy()
    full_data["GDP_growth_lag1"] = full_data[target_col].shift(1)
    full_data["GDP_growth_lag2"] = full_data[target_col].shift(2)
    if "covid_dummy" not in full_data.columns:
        full_data["covid_dummy"] = 0

    all_predictors = selected_names + ["GDP_growth_lag1", "GDP_growth_lag2", "covid_dummy"]

    # --- ULTIMATE BUGFIX: Pad md_trans perfectly from min to max date ---
    if not md_trans.empty:
        max_quarter = full_data.index.max()
        if pd.notna(max_quarter):
            full_idx = pd.date_range(start=md_trans.index.min(), end=max_quarter.end_time, freq='MS')
            md_trans = md_trans.reindex(full_idx)

    print("Starting rigorous historical flash simulation... This may take a minute or two.")

    for i in range(min_train_size, len(full_data)):
        target_quarter = full_data.index[i]
        target_q_str = str(target_quarter).replace("Q", " Q")
        
        # --- NEW LOGICAL BOUNDARY ---
        # If FRED hasn't released the actual GDP for this quarter yet, skip the historical simulation!
        if pd.isna(full_data.loc[target_quarter, target_col]):
            print(f"  [-] Skipping {target_q_str}: No Actual GDP data released by FRED yet.")
            continue
        
        # Training data strictly UP TO the previous quarter
        train_data = full_data.iloc[:i].copy().dropna(subset=[target_col])
        hist_md_end = target_quarter.start_time - pd.Timedelta(days=1)
        train_md = md_trans.loc[:hist_md_end].copy()
        
        try:
            # 2. Train the Models
            bridge_model, _ = fit_bridge_model(train_data, selected_names, target_col=target_col)
            ar_models = fit_ar_models(train_md, selected_names)
        except Exception as e:
            print(f"  [X] Skipped {target_q_str} entirely: Model training failed ({e})")
            continue

        # 3. Simulate Flash 1, Flash 2, and Flash 3 independently
        for flash in [1, 2, 3]:
            try:
                md_flash = _make_flash_monthly_panel(md_trans.copy(), target_quarter, flash, selected_names)
                md_filled = fill_ragged_edge(md_flash, ar_models, selected_names)
                monthly_q_filled = aggregate_to_quarterly(md_filled)
                
                if not isinstance(monthly_q_filled.index, pd.PeriodIndex):
                    monthly_q_filled.index = monthly_q_filled.index.to_period('Q')
                
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
                
                x_forecast = sm.add_constant(x_forecast, has_constant='add')
                x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)
                
                if x_forecast.isna().any().any():
                    missing_cols = x_forecast.columns[x_forecast.isna().any()].tolist()
                    print(f"  [!] Skipped {target_q_str} Flash {flash} due to missing data: {missing_cols}")
                    continue
                    
                pred = float(np.squeeze(bridge_model.predict(x_forecast).values))
                results.append({"target_quarter": target_q_str, "nowcast_month": flash, "prediction": pred})
                
            except Exception as e:
                print(f"  [X] CRITICAL ERROR on {target_q_str} Flash {flash}: {e}")
                
        # 4. "Month After" (Flash 4) - Protected in its own block!
        try:
            x_final = full_data.loc[[target_quarter], all_predictors].copy()
            x_final = sm.add_constant(x_final, has_constant="add")
            x_final = x_final.reindex(columns=bridge_model.model.exog_names)
            
            if not x_final.isna().any().any():
                final_pred = float(np.squeeze(bridge_model.predict(x_final).values))
                results.append({"target_quarter": target_q_str, "nowcast_month": 4, "prediction": final_pred})
            else:
                missing_cols = x_final.columns[x_final.isna().any()].tolist()
                print(f"  [!] Skipped {target_q_str} Flash 4 due to missing data: {missing_cols}")
        except Exception as e:
            print(f"  [X] CRITICAL ERROR on {target_q_str} Flash 4: {e}")

    df_evolution = pd.DataFrame(results)
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