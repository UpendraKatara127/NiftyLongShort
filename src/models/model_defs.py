from sklearn.linear_model import Ridge
from xgboost import XGBClassifier, XGBRegressor

from src.utils import config


def build_regression_model():
    params = config.MODEL_CONFIG["regression"]
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
    )


def build_classification_model():
    params = config.MODEL_CONFIG["classification"]
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
        use_label_encoder=False,
    )


def build_ridge_model():
    return Ridge(alpha=config.RIDGE_ALPHA)
