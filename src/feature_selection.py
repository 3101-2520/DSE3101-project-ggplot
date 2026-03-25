
from config import *
from statsmodels.stats.outliers_influence import variance_inflation_factor

# This file should contain only:
# standardising predictors
# running LASSO / rlasso
# selecting non-zero features
# optional economic screening helpers

def select_features_rlasso(data, target_col='GDP_growth', exclude_cols=None, threshold=1e-6):
    """
    Run rlasso from hdmpy, extract non‑zero coefficients,
    and return the names of selected variables.
    """

    if exclude_cols is None:
        exclude_cols = []

    data = data.dropna().copy()  # drop NA here because LASSO cannot handle missing values 

    cols_to_drop = [target_col] + [col for col in exclude_cols if col in data.columns]
    X_df = data.drop(columns=cols_to_drop)

    # drop constant / zero-variance columns before scaling 
    non_constant_cols = X_df.columns[X_df.std(ddof=0)>0]
    dropped_constant_cols = list(set(X_df.columns) - set(non_constant_cols))
    X_df = X_df[non_constant_cols]

    if dropped_constant_cols:
        print("Dropped constant columns before LASSO:", dropped_constant_cols)

    y = data[target_col].values
    feature_names = X_df.columns.to_numpy()
    X = X_df.values 


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rlasso_result = hdmpy.rlasso(X_scaled, y, post=True)
    
    # Coefficients are stored in rlasso_result.est['coefficients'] as a DataFrame
    coefs_df = rlasso_result.est['coefficients']
    coefs_all = coefs_df.values.flatten()
    
    # If the length includes intercept (should be n_features + 1), drop it
    if len(coefs_all) == X.shape[1] + 1:
        coefs = coefs_all[1:]
    else:
        coefs = coefs_all
    
    selection_summary = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    })

    selection_summary = selection_summary.loc[
        selection_summary["coefficient"].abs() > threshold
    ].copy()

    selection_summary = selection_summary.sort_values(
        by="coefficient",
        key=lambda s: s.abs(),
        ascending=False
    )
    print("Selected variables:", list(selection_summary["feature"]))
    print("Number of selected variables:", len(selection_summary))

    return selection_summary



## calculate VIF for selected features 
def calculate_vif(data, features): 
    X = data[features].dropna().copy()
    
    vif_df = pd.DataFrame()
    vif_df['feature'] = X.columns
    vif_df['VIF'] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])]
    
    return vif_df.sort_values(by="VIF", ascending=False)



## Pairwise correlation check 
def get_high_correlation_pairs(data, features, threshold=0.8):
    """
    Return pairs of selected features with high absolute correlation.
    """
    X = data[features].dropna().copy()

    corr_matrix = X.corr().abs()

    pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if pd.notna(corr_val) and corr_val > threshold:
                pairs.append({
                    'feature_1': cols[i],
                    'feature_2': cols[j],
                    'abs_corr': corr_val
                })
    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "abs_corr"])

    return pd.DataFrame(pairs).sort_values(by='abs_corr', ascending=False)

