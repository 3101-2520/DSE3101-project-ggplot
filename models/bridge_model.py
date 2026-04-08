from config import *


def fit_bridge_model(data, selected_names, target_col='GDP_growth',
                     include_gdp_lags=False, verbose=VERBOSE):
    """
    Fit OLS bridge equation:
    Baseline: GDP_growth ~ selected indicators
    Optional: GDP_growth ~ selected indicators + GDP_growth_lag1 + GDP_growth_lag2
    Returns the fitted model and a dictionary of coefficients.
    """
    mdata = data.copy()

    if include_gdp_lags:
        mdata["GDP_growth_lag1"] = mdata[target_col].shift(1)
        mdata["GDP_growth_lag2"] = mdata[target_col].shift(2)
        regressors = selected_names + ["GDP_growth_lag1", "GDP_growth_lag2"]
    else:
        regressors = selected_names

    mdata = mdata[regressors + [target_col]].copy()
    mdata = mdata.replace([np.inf, -np.inf], np.nan).dropna()

    X_sel = mdata[regressors]
    X_sel = sm.add_constant(X_sel, has_constant='add')
    y = mdata[target_col]

    model = sm.OLS(y, X_sel).fit()

    if verbose:
        print(model.summary())

    coef_dict = {'intercept': model.params['const']}
    for name in regressors:
        coef_dict[name] = model.params[name]

    return model, coef_dict