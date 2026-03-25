
from config import *

def fit_bridge_model(data, selected_names, target_col='GDP_growth', verbose = VERBOSE):
    """
    Fit OLS bridge equation: GDP_growth ~ selected indicators (quarterly aggregates).
    Returns the fitted model and a dictionary of coefficients.
    """
    mdata = data[selected_names + [target_col]].copy()

    mdata = mdata.replace([np.inf, -np.inf], np.nan).dropna()

    X_sel = mdata[selected_names]
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