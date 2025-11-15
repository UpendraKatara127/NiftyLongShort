from datetime import time
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")
TRADING_START = time(9, 15)
TRADING_END = time(15, 30)
HOURLY_INTERVALS = [
    (time(9, 15), time(10, 15), time(10, 15)),
    (time(10, 15), time(11, 15), time(11, 15)),
    (time(11, 15), time(12, 15), time(12, 15)),
    (time(12, 15), time(13, 15), time(13, 15)),
    (time(13, 15), time(14, 15), time(14, 15)),
]
TRAIN_START = "2021-04-01"
TRAIN_END = "2025-03-31"
TEST_START = "2024-04-01"
TEST_END = "2025-03-31"
LOOKBACK_DAYS = 200
WALKFORWARD_STEP_DAYS = 1
PCA_WINDOW = 200
BETA_WINDOW = 120
REGIME_WINDOW = 250
VOL_LOOKBACK = 60
VOL_OF_VOL_LOOKBACK = 90
TIME_SERIES_ZSCORE_WINDOW = 120
EXPOSURE_SCALE_RANGE = (0.5, 1.0)
TRANSACTION_COST = 0.001
CAPITAL = 10_000_000
LONG_EXPOSURE = 1.0
SHORT_EXPOSURE = 1.0
TOP_QUANTILE = 0.1
MAX_BASKET_SIZE = 3
MIN_SCORE_Z = 1.0
MAX_WEIGHT_PER_STOCK = 0.1
TURNOVER_LIMIT = 1.5
PIPELINE_CLIP = (0.01, 0.99)
FEATURE_IMPUTE_VALUE = 0.0
RISK_FREE_RATE = 0.05
MODEL_CONFIG = {
    "regression": {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
    },
    "classification": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
    },
}
ALPHA_STACK = 0.5
SCORE_ENTROPY_MAX = 0.95
