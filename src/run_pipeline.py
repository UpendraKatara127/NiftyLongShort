import pandas as pd

from src.backtest.run_backtest import run_backtest
from src.data_prep.build_panel import build_hourly_panel
from src.models.train_walkforward import train_walkforward
from src.utils import paths


def run_all(force_rebuild=True, force_retrain=True):
    if force_rebuild or not paths.HOURLY_PANEL_PATH.exists():
        print("Building hourly panel...")
        panel = build_hourly_panel()
        print(f"Hourly panel built with {len(panel)} rows")
    else:
        print("Hourly panel already exists, reusing cached parquet.")
    if force_retrain or not paths.SIGNALS_PATH.exists():
        print("Training walk-forward models and generating signals...")
        signals = train_walkforward()
        print(f"Signals generated for {len(signals)} observations")
    else:
        print("Signals already exist, skipping retraining.")
        try:
            existing = pd.read_parquet(paths.SIGNALS_PATH, columns=["score"])
            print(f"Existing signals found with {len(existing)} rows")
        except Exception:
            existing = None
    print("Running backtest...")
    trade_log, equity, metrics = run_backtest()
    print(f"Backtest completed with {len(trade_log)} trades and {len(equity)} equity points")
    print("Performance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_all()
