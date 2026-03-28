
from config import *
from src.data_preprocessing import load_and_transform_md, aggregate_to_quarterly, load_and_transform_qd, merge_data
from src.feature_selection import select_features_rlasso, get_high_correlation_pairs
from models.ar_indicator import fit_ar_models, fill_ragged_edge
from models.bridge_model import fit_bridge_model
from models.evaluation import run_rolling_nowcast, run_rf_benchmark
from models.ar_benchmark import run_ar_benchmark
from models.adl_benchmark import run_adl_benchmark
from models.flash_nowcast import run_rolling_flash_nowcast
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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

    # Add COVID dummy
    data['covid_dummy'] = 0
    data.loc[(data.index >= pd.Period('2020Q1', freq='Q')) & 
            (data.index <= pd.Period('2020Q2', freq='Q')), 'covid_dummy'] = 1

    # Step 5: Train/test split (keep last 8 quarters for testing)
    test_size = 8
    train_data = data.iloc[:-test_size].copy()
    test_data = data.iloc[-test_size:].copy()

    print(f"Training sample: {train_data.index.min()} to {train_data.index.max()}, shape = {train_data.shape}") 
    print(f"Testing sample: {test_data.index.min()} to {test_data.index.max()}, shape = {test_data.shape}")

    # Step 6: Feature selection with rlasso on training data
    selected_summary = select_features_rlasso(train_data, target_col='GDP_growth', exclude_cols = ['covid_dummy'])
    selected = list(selected_summary["feature"])

    # Step 7: Check for high correlation among selected features
    high_corr_pairs = get_high_correlation_pairs(data, selected, threshold=0.9)
    if not high_corr_pairs.empty:
        print("\nWarning: High correlation detected among selected features:")
        print(high_corr_pairs)
    else:
        print("\nNo high correlation detected among selected features.")
    
    # Step 8: Fit bridge equation (OLS) using selected variables on training data
    bridge_model, bridge_coefs = fit_bridge_model(train_data, selected)
    
    # Step 9: Fit AR(p) models for each selected indicator (for ragged‑edge forecasting)
    ar_models = fit_ar_models(MD_trans, selected, max_lag=12)

    # Step 10: Fill the ragged edge of the monthly indicators using the fitted AR models
    MD_filled = fill_ragged_edge(MD_trans, ar_models, selected)
    
    print("\nSelected variables before AR filling (last 8 rows):")
    print(MD_trans[selected].tail(8))
    print("\nSelected variables after AR filling (last 8 rows):")
    print(MD_filled[selected].tail(8))

    # Step 11: Re-aggregate filled monthly data to quarterly
    monthly_q_filled = aggregate_to_quarterly(MD_filled)

    print("\nFilled quarterly selected variables (last 4 quarters):")
    print(monthly_q_filled[selected].tail(4))

    print("\nAll preprocessing and model fitting complete.")
    print("You can now use the selected variables, bridge coefficients, and AR models for nowcasting.")

    # Step 12: Rolling window evaluation (optimal window size = 180)
    test_size = 8
    window_size = 180
    rolling_results = run_rolling_nowcast(
        data, MD_trans, selected,
        test_size=test_size,
        window_size=window_size,
        max_lag=12,
        target_col='GDP_growth',
        verbose=VERBOSE
    )

    # Step 12: Rolling window evaluation with fixed window sizes (grid search)
    """
    test_size = 8                     # last 8 quarters (2018Q3–2020Q2)
    # List of window sizes to test (quarters)
    window_sizes = [60, 80, 100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 234]

    for ws in window_sizes:
        print(f"\n{'='*60}")
        print(f"Testing window size = {ws} quarters")
        print('='*60)
        run_rolling_nowcast(
            data, MD_trans, selected,
            test_size=test_size,
            window_size=ws,
            max_lag=12,
            target_col='GDP_growth',
            verbose=VERBOSE
        )
    """
    # Step 13: Run flash nowcast evaluation
    flash_results = run_rolling_flash_nowcast(
        data=data,
        md_trans=MD_trans,
        selected=selected,
        test_size=test_size,
        window_size=80,
        max_lag=12,
        target_col='GDP_growth',
        flashes=(1, 2, 3),
        verbose=VERBOSE
    )

    # Step 13: Run AR benchmark evaluation
    ar_benchmark_results = run_ar_benchmark(data, test_size=test_size, target_col='GDP_growth', max_lag=8)

    # Step 14: Run ADL benchmark evaluation
    adl_benchmark_results = run_adl_benchmark(data, test_size=test_size, target_col='GDP_growth')

    # Step 15: Run RF benchmark evaluation

    X_train = train_data[selected]
    y_train = train_data['GDP_growth']
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf_params = grid_search.best_params_
    print("\nBest RF parameters found:")
    print(best_rf_params)

    rf_results = run_rf_benchmark(
        data=data,
        selected=selected,
        test_size=test_size,
        target_col="GDP_growth",
        rf_params=best_rf_params
    )
