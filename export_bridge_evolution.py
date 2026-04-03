import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys

# Ensure the script can find your src and models folders
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.bridge_model import fit_bridge_model
from models.ar_indicator import fit_ar_models, fill_ragged_edge
from models.flash_nowcast import _make_flash_monthly_panel, _build_flash_predictor_row
from src.data_preprocessing import load_and_transform_md, load_and_transform_qd, aggregate_to_quarterly, merge_data
from frontend.components.atlanta_fed import annualize_gdp_growth

def build_bridge_evolution_csv(data, md_trans, selected_names, output_path=None, target_col='GDP_growth', min_train_size=20):
    if output_path is None:
        output_path = ROOT / "data" / "bridge_evolution.csv"

    results = []
    
    # 1. Prepare full dataset with the correct lags and dummies
    full_data = data.copy()
    full_data["GDP_growth_lag1"] = full_data[target_col].shift(1)
    full_data["GDP_growth_lag2"] = full_data[target_col].shift(2)
    if "covid_dummy" not in full_data.columns:
        full_data["covid_dummy"] = 0

    all_predictors = selected_names + ["GDP_growth_lag1", "GDP_growth_lag2", "covid_dummy"]

    print("Starting rigorous historical flash simulation... This may take a minute or two.")

    for i in range(min_train_size, len(full_data)):
        target_quarter = full_data.index[i]
        target_q_str = str(target_quarter).replace("Q", " Q")
        
        # Training data strictly UP TO the previous quarter
        train_data = full_data.iloc[:i].copy()
        train_data = train_data.dropna(subset=[target_col])
        
        # Historical monthly data available up to the start of the target quarter
        hist_md_end = target_quarter.start_time - pd.Timedelta(days=1)
        train_md = md_trans.loc[:hist_md_end].copy()
        
        try:
            # 2. Train the Bridge Model on historical data
            bridge_model, _ = fit_bridge_model(train_data, selected_names, target_col=target_col)
            
            # 3. Train the AR models on historical monthly data
            ar_models = fit_ar_models(train_md, selected_names)
            
            # 4. Simulate Flash 1, Flash 2, and Flash 3 (The true "Ragged Edge")
            for flash in [1, 2, 3]:
                # Hide the 'future' months of the quarter
                md_flash = _make_flash_monthly_panel(md_trans.copy(), target_quarter, flash, selected_names)
                
                # Fill missing edge data using AR models
                md_filled = fill_ragged_edge(md_flash, ar_models, selected_names)
                monthly_q_filled = aggregate_to_quarterly(md_filled)
                
                if not isinstance(monthly_q_filled.index, pd.PeriodIndex):
                    monthly_q_filled.index = monthly_q_filled.index.to_period('Q')
                
                # Build the prediction row for this flash
                x_forecast = _build_flash_predictor_row(
                    monthly_q_filled=monthly_q_filled,
                    data=full_data,
                    forecast_quarter=target_quarter,
                    selected=selected_names,
                    target_col=target_col
                )
                
                # Add lagging and dummy variables
                x_forecast["GDP_growth_lag1"] = full_data.loc[target_quarter - 1, target_col]
                x_forecast["GDP_growth_lag2"] = full_data.loc[target_quarter - 2, target_col]
                x_forecast["covid_dummy"] = full_data.loc[target_quarter, "covid_dummy"]
                
                x_forecast = sm.add_constant(x_forecast, has_constant='add')
                x_forecast = x_forecast.reindex(columns=bridge_model.model.exog_names)
                
                if x_forecast.isna().any().any():
                    continue
                    
                pred = bridge_model.predict(x_forecast).iloc[0]
                results.append({"target_quarter": target_q_str, "nowcast_month": flash, "prediction": pred})
                
            # 5. "Month After" (Flash 4) - All data is finally published!
            x_final = full_data.loc[[target_quarter], all_predictors].copy()
            x_final = sm.add_constant(x_final, has_constant="add")
            x_final = x_final.reindex(columns=bridge_model.model.exog_names)
            
            if not x_final.isna().any().any():
                final_pred = bridge_model.predict(x_final).iloc[0]
                results.append({"target_quarter": target_q_str, "nowcast_month": 4, "prediction": final_pred})
            
        except Exception:
            continue

    df_evolution = pd.DataFrame(results)
    if output_path is not None:
        df_evolution.to_csv(output_path, index=False)
        print(f"\n✅ Successfully exported accurate evolution data to {output_path}")
        
    return df_evolution

# ==========================================
# EXECUTION BLOCK (This actually runs it!)
# ==========================================
if __name__ == "__main__":
    print("Loading datasets...")
    md_path = ROOT / "data" / "2026-02-MD.csv"
    qd_path = ROOT / "data" / "2026-02-QD.csv"
    output_path = ROOT / "data" / "bridge_evolution.csv"

    # 1. Load and transform Monthly Data
    MD_trans = load_and_transform_md(md_path)
    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

    # 2. Load and transform Quarterly GDP
    GDP_growth = load_and_transform_qd(qd_path, gdp_col="GDPC1")
    GDP_growth = annualize_gdp_growth(GDP_growth)
    GDP_growth.name = "GDP_growth"

    # 3. Aggregate to Quarterly & Merge
    monthly_q = aggregate_to_quarterly(MD_trans)
    data, _, _ = merge_data(monthly_q, GDP_growth)

    # 4. Add COVID dummy
    data["covid_dummy"] = 0
    data.loc[
        (data.index >= pd.Period("2020Q1", freq="Q")) &
        (data.index <= pd.Period("2020Q2", freq="Q")),
        "covid_dummy"
    ] = 1

    # Your selected bridge variables
    selected = ["IPDMAT", "DPCERA3M086SBEA", "PAYEMS", "UEMP15T26", "PERMITNE", "UNRATE", "HWIURATIO"]

    # RUN THE SIMULATION
    build_bridge_evolution_csv(
        data=data,
        md_trans=MD_trans,
        selected_names=selected,
        output_path=output_path,
        target_col="GDP_growth",
        min_train_size=20
    )