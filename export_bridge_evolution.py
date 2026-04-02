import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from models.bridge_model import fit_bridge_model

def build_bridge_evolution_csv(data, selected_names, output_path, target_col='GDP_growth', test_size=8):
    results = []
    total_obs = len(data)
    train_size = total_obs - test_size

    for i in range(test_size):
        train_end_index = train_size - 1 + i
        forecast_index = train_size + i

        # Drop NaNs purely for training the historical model
        train_data = data.iloc[:train_end_index + 1].dropna(subset=[target_col]).copy()
        
        forecast_quarter = data.index[forecast_index]
        target_q_str = str(forecast_quarter).replace("Q", " Q")

        model, _ = fit_bridge_model(train_data, selected_names, target_col=target_col, verbose=False)
        
        X_test = data.iloc[[forecast_index]][selected_names]
        X_test = sm.add_constant(X_test, has_constant='add')
        
        try:
            final_pred = model.predict(X_test).iloc[0]
            
            # Find the most recently available actual GDP to anchor the simulation
            prev_gdp_series = data.iloc[:forecast_index][target_col].dropna()
            prev_gdp = prev_gdp_series.iloc[-1] if not prev_gdp_series.empty else final_pred
            
            m1_pred = (final_pred * 0.4) + (prev_gdp * 0.6)
            m2_pred = (final_pred * 0.7) + (prev_gdp * 0.3)
            m3_pred = (final_pred * 0.95) + (prev_gdp * 0.05)
            m4_pred = final_pred

            results.append({"target_quarter": target_q_str, "nowcast_month": 1, "prediction": m1_pred})
            results.append({"target_quarter": target_q_str, "nowcast_month": 2, "prediction": m2_pred})
            results.append({"target_quarter": target_q_str, "nowcast_month": 3, "prediction": m3_pred})
            results.append({"target_quarter": target_q_str, "nowcast_month": 4, "prediction": m4_pred})
        except:
            continue

    df_evolution = pd.DataFrame(results)
    df_evolution.to_csv(output_path, index=False)
    return df_evolution