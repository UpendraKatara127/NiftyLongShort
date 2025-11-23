from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .costs import TransactionCostModel
from .data import MinuteBarDataLoader
from .metrics import (
    compute_daily_pnl,
    compute_hourly_pnl,
    compute_summary_stats,
)
from .portfolio import PortfolioBook
from .strategy import BaseStrategy


@dataclass
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: pd.Series
    hourly_pnl: pd.Series
    daily_pnl: pd.Series
    summary_stats: dict


class BacktestEngine:
    def __init__(
        self,
        data_loader: MinuteBarDataLoader,
        strategy: BaseStrategy,
        start_date: str,
        end_date: str,
        rebalance_times: List[str],
    ):
        self.data_loader = data_loader
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_times = rebalance_times

    def run(self) -> BacktestResult:
        data = self.data_loader.load_data(self.start_date, self.end_date)
        rebalance_data = self.data_loader.get_rebalance_data(
            self.rebalance_times, self.start_date, self.end_date
        )
        if rebalance_data.empty:
            raise ValueError("No rebalance data available for the given date range.")
        rebalance_data = (
            rebalance_data.groupby(level=["datetime", "symbol"]).last()
        )
        price_panel = (
            rebalance_data.reset_index()
            .pivot(index="datetime", columns="symbol", values="close")
            .sort_index()
        )
        price_panel = price_panel.ffill()
        timestamps = price_panel.index.to_list()
        self.strategy.on_backtest_start(data)
        cost_model = TransactionCostModel(self.strategy.cost_bps_per_side)
        portfolio = PortfolioBook(self.strategy.capital, cost_model)
        stop_loss_pct = getattr(self.strategy, "stop_loss_pct", None)
        equity_points = {}
        for idx, timestamp in enumerate(timestamps):
            if stop_loss_pct is not None:
                self._apply_stop_losses(portfolio, data, timestamp, stop_loss_pct)
            current_prices = price_panel.loc[timestamp]
            portfolio.close_all(timestamp, current_prices)
            equity_points[timestamp] = portfolio.equity
            history_slice = self.data_loader.get_history_slice(
                timestamp, self.start_date, self.end_date
            )
            if not self._should_open_new_positions(idx, timestamps):
                continue
            target_weights = self.strategy.generate_target_weights(timestamp, history_slice)
            portfolio.open_positions(timestamp, target_weights, current_prices)
        equity_curve = pd.Series(equity_points).sort_index()
        columns = [
            "timestamp",
            "symbol",
            "side",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl",
            "pnl_pct",
        ]
        trade_log = pd.DataFrame(portfolio.trade_log)
        if trade_log.empty:
            trade_log = pd.DataFrame(columns=columns)
        else:
            trade_log = (
                trade_log[columns].sort_values("timestamp").reset_index(drop=True)
            )
        hourly_pnl = compute_hourly_pnl(equity_curve)
        daily_pnl = compute_daily_pnl(equity_curve)
        summary_stats = compute_summary_stats(
            equity_curve,
            trade_log,
            portfolio.initial_capital,
            portfolio.daily_turnover,
        )
        result = BacktestResult(
            trade_log=trade_log,
            equity_curve=equity_curve,
            hourly_pnl=hourly_pnl,
            daily_pnl=daily_pnl,
            summary_stats=summary_stats,
        )
        self.strategy.on_backtest_end(result.summary_stats)
        return result

    def _should_open_new_positions(self, idx: int, timestamps: List[pd.Timestamp]) -> bool:
        if idx >= len(timestamps) - 1:
            return False
        current = timestamps[idx]
        nxt = timestamps[idx + 1]
        return current.date() == nxt.date()


    def _apply_stop_losses(
        self,
        portfolio: PortfolioBook,
        data: pd.DataFrame,
        timestamp: pd.Timestamp,
        stop_loss_pct: float,
    ) -> None:
        if stop_loss_pct <= 0 or not portfolio.positions:
            return
        for symbol, position in list(portfolio.positions.items()):
            if timestamp <= position.entry_timestamp:
                continue
            try:
                symbol_frame = data.xs(symbol, level="symbol")
            except KeyError:
                continue
            mask = (symbol_frame.index > position.entry_timestamp) & (
                symbol_frame.index <= timestamp
            )
            window = symbol_frame.loc[mask]
            if window.empty:
                continue
            if position.quantity > 0:
                stop_price = position.entry_price * (1 - stop_loss_pct)
                breach = window[window["low"] <= stop_price]
            else:
                stop_price = position.entry_price * (1 + stop_loss_pct)
                breach = window[window["high"] >= stop_price]
            if breach.empty:
                continue
            stop_time = breach.index[0]
            portfolio.close_position_manual(symbol, stop_price, stop_time)
