from config import *
from src.data_preprocessing import load_and_transform_md, aggregate_to_quarterly, load_and_transform_qd, merge_data
from src.feature_selection import select_features_rlasso, get_high_correlation_pairs
from models.ar_indicator import fit_ar_models
from models.bridge_model import fit_bridge_model
from models.evaluation import run_rolling_nowcast   # <-- import the evaluation function

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # File paths (adjust if needed)
    ROOT = Path(__file__).resolve().parents[1]
    md_path = ROOT / "data/2026-02-MD.csv"
    qd_path = ROOT / "data/2026-02-QD.csv"
    
    # Step 1: Load and transform monthly data
    MD_trans = load_and_transform_md(md_path)

    # Drop unwanted variables
    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')
    print(f"Dropped {len(vars_to_drop)} variables. New shape: {MD_trans.shape}")
    
    # Step 2: Load and transform quarterly GDP
    GDP_growth = load_and_transform_qd(qd_path, gdp_col='GDPC1')

    start_period = pd.Period('1960Q1', freq='Q')
    end_period = pd.Period('2020Q2', freq='Q')
    start_date = start_period.start_time
    end_date = end_period.end_time
    MD_trans = MD_trans.loc[start_date:end_date]
    print(f"Filtered MD_trans to {start_date.date()} – {end_date.date()}. New shape: {MD_trans.shape}")
    GDP_growth = GDP_growth.loc[start_period:end_period]
    print(f"Filtered GDP_growth to {start_period} – {end_period}. Length: {len(GDP_growth)}")
    
    # Step 3: Aggregate monthly indicators to quarterly
    monthly_q = aggregate_to_quarterly(MD_trans)
    
    # Step 4: Merge with GDP growth
    data, X, y = merge_data(monthly_q, GDP_growth)

    # Step 5: Feature selection with rlasso
    feature_names = data.drop(columns=['GDP_growth']).columns
    selected_summary = select_features_rlasso(data, target_col='GDP_growth')
    selected = list(selected_summary["feature"])

    # Step 6: Check for high correlation among selected features
    high_corr_pairs = get_high_correlation_pairs(data, selected, threshold=0.9)
    if not high_corr_pairs.empty:
        print("\nWarning: High correlation detected among selected features:")
        print(high_corr_pairs)
    else:
        print("\nNo high correlation detected among selected features.")
    
    # Step 7: Fit bridge equation (OLS) using selected variables (full sample)
    bridge_model, bridge_coefs = fit_bridge_model(data, selected)
    
    # Step 8: Fit AR(p) models for each selected indicator (for ragged‑edge forecasting)
    ar_models = fit_ar_models(MD_trans, selected, max_lag=12)
    
    print("\nAll preprocessing and model fitting complete.")
    print("You can now use the selected variables, bridge coefficients, and AR models for nowcasting.")

    # ------------------------------------------------------------------
    # Rolling window evaluation with fixed window size (80 quarters)
    # ------------------------------------------------------------------
    test_size = 8                     # last 8 quarters (2018Q3–2020Q2)
    window_size = 234                  # fixed window length (20 years)
    rolling_results = run_rolling_nowcast(
        data, MD_trans, selected,
        test_size=test_size,
        window_size=window_size,
        max_lag=12,
        target_col='GDP_growth',
        verbose=VERBOSE               # use the flag from config
    )