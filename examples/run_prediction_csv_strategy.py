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


class PredictionCSVStrategy(BaseStrategy):
    def __init__(
        self,
        capital: float,
        cost_bps_per_side: float,
        predictions_csv: str,
        basket_size: int = 5,
        rebalance_time: str = "10:15",
        timezone: str = "Asia/Kolkata",
        stop_loss_pct: float = 0.0,
    ):
        super().__init__(
            capital=capital,
            cost_bps_per_side=cost_bps_per_side,
            stop_loss_pct=stop_loss_pct,
        )
        self.basket_size = basket_size
        self.rebalance_time = rebalance_time
        self.timezone = timezone
        self.predictions = self._load_predictions(predictions_csv)

    def _load_predictions(self, predictions_csv: str) -> pd.DataFrame:
        df = pd.read_csv(predictions_csv)
        required_cols = {"ticker", "date", "y_pred"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Prediction file missing columns: {missing}")
        df["timestamp"] = pd.to_datetime(df["date"] + f" {self.rebalance_time}")
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.timezone)
        df = (
            df[["timestamp", "ticker", "y_pred"]]
            .pivot(index="timestamp", columns="ticker", values="y_pred")
            .sort_index()
        )
        return df

    def generate_target_weights(
        self, timestamp: pd.Timestamp, prices_slice: pd.DataFrame
    ) -> dict:
        if timestamp.strftime("%H:%M") != self.rebalance_time:
            return {}
        try:
            preds = self.predictions.loc[timestamp].dropna()
        except KeyError:
            return {}
        if preds.empty:
            return {}
        n_bucket = min(self.basket_size, len(preds))
        long_symbols = preds.nlargest(n_bucket).index
        print("Long symbols:", long_symbols.tolist())
        short_symbols = preds.nsmallest(n_bucket).index
        print("Short symbols:", short_symbols.tolist())
        gross_exposure = 1.0
        weight_per_side = (gross_exposure / 2) / n_bucket if n_bucket else 0.0
        weights = {symbol: weight_per_side for symbol in long_symbols}
        weights.update({symbol: -weight_per_side for symbol in short_symbols})
        return weights


def main():
    repo_root = ROOT
    data_root = repo_root / "data"
    loader = MinuteBarDataLoader(
        data_root=str(data_root),
        universe_csv=str(data_root / "nifty_50_constituents.csv"),
    )
    strategy = PredictionCSVStrategy(
        capital=10_00_000,
        cost_bps_per_side=10,
        predictions_csv=str(repo_root / "predictions_for_1115_at_1015.csv"),
        basket_size=1,
        rebalance_time="10:15",
    )
    engine = BacktestEngine(
        data_loader=loader,
        strategy=strategy,
        start_date="2019-12-31",
        end_date="2021-04-05",
        rebalance_times=["10:15", "11:15", "12:15", "13:15", "14:15"],
    )
    print("Running backtest...")
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

    results.trade_log.to_csv("trade_log.csv", index=False)


if __name__ == "__main__":
    main()
