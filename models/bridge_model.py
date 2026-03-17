
from scripts.config import *






def fit_bridge_model(data, selected_names):
    """
    Fit OLS bridge equation: GDP_growth ~ selected indicators (quarterly aggregates).
    Returns the fitted model and a dictionary of coefficients.
    """
    X_sel = data[selected_names]
    X_sel = sm.add_constant(X_sel)
    y = data['GDP_growth']
    model = sm.OLS(y, X_sel).fit()
    print(model.summary())
    
    # Store coefficients
    coef_dict = {'intercept': model.params['const']}
    for name in selected_names:
        coef_dict[name] = model.params[name]
    return model, coef_dict