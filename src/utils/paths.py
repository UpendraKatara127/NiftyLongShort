from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MINUTE_DATA_DIR = DATA_DIR / "minute_wise"
DAY_DATA_DIR = DATA_DIR / "day_wise"
CONSTITUENTS_PATH = DATA_DIR / "nifty_50_constituents.csv"
SECTOR_PATH = DATA_DIR / "nifty_50_sectors.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs"
HOURLY_PANEL_PATH = OUTPUTS_DIR / "hourly_panel.parquet"
SIGNALS_PATH = OUTPUTS_DIR / "signals.parquet"
FEATURE_PIPELINE_DIR = OUTPUTS_DIR / "feature_pipelines"
MODELS_DIR = OUTPUTS_DIR / "models"
BACKTEST_DIR = OUTPUTS_DIR / "backtests"
EQUITY_CURVE_PATH = BACKTEST_DIR / "equity_curve.png"
PERFORMANCE_PATH = BACKTEST_DIR / "performance.json"
TRADE_LOG_PATH = BACKTEST_DIR / "trade_log.csv"
