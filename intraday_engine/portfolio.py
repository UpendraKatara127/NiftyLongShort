from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .costs import TransactionCostModel


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    side: str
    entry_cost: float


class PortfolioBook:
    def __init__(self, capital: float, cost_model: TransactionCostModel):
        self.initial_capital = capital
        self.equity = capital
        self.cost_model = cost_model
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[dict] = []
        self.daily_turnover: Dict[pd.Timestamp, float] = {}

    def open_positions(
        self,
        timestamp: pd.Timestamp,
        target_weights: Dict[str, float],
        prices: pd.Series,
    ) -> None:
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or pd.isna(price) or price <= 0:
                continue
            quantity = (weight * self.equity) / price
            if quantity == 0:
                continue
            notional = quantity * price
            entry_cost = self.cost_model.cost(notional)
            self.equity -= entry_cost
            side = "buy" if quantity > 0 else "sell"
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_timestamp=timestamp,
                side=side,
                entry_cost=entry_cost,
            )
            self._accumulate_turnover(timestamp, abs(notional))

    def close_all(self, timestamp: pd.Timestamp, prices: pd.Series) -> List[dict]:
        closed_trades = []
        for symbol, position in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None or pd.isna(price):
                continue
            trade = self._close_position(symbol, position, price, timestamp)
            if trade:
                closed_trades.append(trade)
        return closed_trades

    def close_positions(
        self, timestamp: pd.Timestamp, prices: pd.Series, symbols: List[str]
    ) -> List[dict]:
        closed_trades = []
        for symbol in symbols:
            position = self.positions.get(symbol)
            if position is None:
                continue
            price = prices.get(symbol)
            if price is None or pd.isna(price):
                continue
            trade = self._close_position(symbol, position, price, timestamp)
            if trade:
                closed_trades.append(trade)
        return closed_trades

    def close_position_manual(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp,
    ) -> dict:
        if price is None or pd.isna(price):
            return None
        position = self.positions.get(symbol)
        if position is None:
            return None
        return self._close_position(symbol, position, price, timestamp)

    def _close_position(
        self,
        symbol: str,
        position: Position,
        price: float,
        timestamp: pd.Timestamp,
    ) -> dict:
        notional = position.quantity * price
        exit_cost = self.cost_model.cost(notional)
        pnl = (price - position.entry_price) * position.quantity
        net_pnl = pnl - exit_cost
        total_cost = position.entry_cost + exit_cost
        self.equity += pnl - exit_cost
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": price,
            "quantity": position.quantity,
            "pnl": net_pnl,
            "pnl_pct": net_pnl
            / abs(position.entry_price * position.quantity)
            if position.quantity != 0
            else 0.0,
            "transaction_cost": total_cost,
        }
        self.trade_log.append(trade)
        self._accumulate_turnover(timestamp, abs(notional))
        del self.positions[symbol]
        return trade

    def _accumulate_turnover(self, timestamp: pd.Timestamp, notional: float) -> None:
        date_key = pd.Timestamp(timestamp.date())
        self.daily_turnover[date_key] = self.daily_turnover.get(date_key, 0.0) + notional
