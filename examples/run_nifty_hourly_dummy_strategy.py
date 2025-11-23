from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intraday_engine.data import MinuteBarDataLoader
from intraday_engine.engine import BacktestEngine
from intraday_engine.strategy import BaseStrategy
from intraday_engine.visualization import (
    plot_daily_pnl,
    plot_drawdown,
    plot_equity_curve,
    plot_hourly_pnl_hist,
)


class MyHourlyDummyStrategy(BaseStrategy):
    def __init__(self, capital: float, cost_bps_per_side: float, basket_size: int = 5):
        super().__init__(capital=capital, cost_bps_per_side=cost_bps_per_side)
        self.basket_size = basket_size

    def generate_target_weights(
        self, timestamp: pd.Timestamp, prices_slice: pd.DataFrame
    ) -> dict:
        close_panel = (
            prices_slice.reset_index()
            .pivot(index="datetime", columns="symbol", values="close")
            .sort_index()
        )
        lookback_bars = 60
        if len(close_panel) < lookback_bars:
            return {}
        # Rolling window mimics the `backtesting.py` style indicator evaluation.
        window = close_panel.tail(lookback_bars)
        momentum_scores = (window.iloc[-1] / window.iloc[0]) - 1.0
        momentum_scores = momentum_scores.dropna()
        if momentum_scores.empty:
            return {}
        # Same percentile selection naming as typical `backtesting.py` recipes.
        n_bucket = min(self.basket_size, len(momentum_scores))
        long_symbols = momentum_scores.nlargest(n_bucket).index
        short_symbols = momentum_scores.nsmallest(n_bucket).index
        gross_exposure = 1.0
        # Allocate half the book to longs and half to shorts, mirroring the
        # gross/net exposure conventions from the `backtesting.py` cookbook.
        weight_per_side = (gross_exposure / 2) / n_bucket if n_bucket else 0.0
        weights = {symbol: weight_per_side for symbol in long_symbols}
        weights.update({symbol: -weight_per_side for symbol in short_symbols})
        # Returned dict matches the engine expectation (symbol -> target weight).
        return weights


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    loader = MinuteBarDataLoader(
        data_root=str(data_root),
        universe_csv=str(data_root / "nifty_50_constituents.csv"),
        market_open="09:15",
        market_close="15:30",
    )
    strategy = MyHourlyDummyStrategy(
        capital=10_000_000,
        cost_bps_per_side=10,
        basket_size=5,
    )
    engine = BacktestEngine(
        data_loader=loader,
        strategy=strategy,
        start_date="2021-04-01",
        end_date="2021-04-30",
        rebalance_times=["10:15", "11:15", "12:15", "13:15", "14:15"],
    )
    results = engine.run()
    print("Summary Stats:")
    for key, value in results.summary_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    plot_equity_curve(results.equity_curve)
    plot_drawdown(results.equity_curve)
    plot_daily_pnl(results.daily_pnl)
    plot_hourly_pnl_hist(results.hourly_pnl)
    plt.show()


if __name__ == "__main__":
    main()
