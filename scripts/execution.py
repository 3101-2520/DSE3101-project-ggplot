
from config import *
from scripts.data_preprocessing import load_and_transform_md, aggregate_to_quarterly, load_and_transform_qd, merge_data
from scripts.feature_selection import select_features_rlasso
from models.ar_indicator import fit_ar_models
from models.bridge_model import fit_bridge_equation

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
    
    # Step 2: Load and transform quarterly GDP
    GDP_growth = load_and_transform_qd(qd_path, gdp_col='GDPC1')
    
    # Step 3: Aggregate monthly indicators to quarterly
    monthly_q = aggregate_to_quarterly(MD_trans)
    
    # Step 4: Merge with GDP growth
    data, X, y = merge_data(monthly_q, GDP_growth)
    
    # Step 5: Feature selection with rlasso
    feature_names = data.drop(columns=['GDP_growth']).columns
    selected = select_features_rlasso(X, y, feature_names)
    
    # Step 6: Fit bridge equation (OLS) using selected variables
    bridge_model, bridge_coefs = fit_bridge_equation(data, selected)
    
    # Step 7: Fit AR(p) models for each selected indicator (for ragged‑edge forecasting)
    ar_models = fit_ar_models(MD_trans, selected, max_lag=12)
    
    print("\nAll preprocessing and model fitting complete.")
    print("You can now use the selected variables, bridge coefficients, and AR models for nowcasting.")