
from config import *

def fit_bridge_model(data, selected_names, target_col='GDP_growth', verbose = VERBOSE):
    """
    Fit OLS bridge equation: GDP_growth ~ selected indicators (quarterly aggregates).
    Returns the fitted model and a dictionary of coefficients.
    """
    mdata = data.copy()

    # Add lagged GDP variables
    mdata['GDP_growth_lag1'] = mdata[target_col].shift(1)
    mdata['GDP_growth_lag2'] = mdata[target_col].shift(2)

    # Keep only required columns and drop rows with missing values
    mdata = mdata[selected_names + [target_col, 'GDP_growth_lag1', 'GDP_growth_lag2', 'covid_dummy']].copy()
    mdata = mdata.replace([np.inf, -np.inf], np.nan).dropna()

    predictors = ['GDP_growth_lag1', 'GDP_growth_lag2'] + selected_names + ['covid_dummy']
    X_sel = mdata[predictors]
    X_sel = sm.add_constant(X_sel)
    y = mdata['GDP_growth']
    model = sm.OLS(y, X_sel).fit()
    if verbose:
        print(model.summary())
    
    # Store coefficients
    coef_dict = {'intercept': model.params['const']}
    for name in selected_names:
        coef_dict[name] = model.params[name]
    return model, coef_dict