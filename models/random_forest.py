from config import *

def fit_rf_model(train_data, feature_cols, target_col='GDP_growth', rf_params=None):
    """
    Fit a low-frequency Random Forest on quarterly data.
    """
    X_train = train_data[feature_cols].copy()
    y_train = train_data[target_col].copy()

    default_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    }

    if rf_params is not None:
        default_params.update(rf_params)

    rf_model = RandomForestRegressor(**default_params)
    rf_model.fit(X_train, y_train)

    return rf_model


def predict_rf_model(rf_model, x_forecast):
    """
    Generate prediction from fitted RF model.
    """
    y_pred = rf_model.predict(x_forecast)[0]
    return y_pred