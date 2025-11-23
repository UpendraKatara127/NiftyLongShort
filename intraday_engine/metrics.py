from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_hourly_pnl(equity_curve: pd.Series) -> pd.Series:
    hourly = equity_curve.diff().fillna(0.0)
    return hourly


def compute_daily_pnl(equity_curve: pd.Series) -> pd.Series:
    daily_marks = equity_curve.groupby(equity_curve.index.date).last()
    daily_series = pd.Series(daily_marks, index=pd.to_datetime(daily_marks.index))
    daily_pnl = daily_series.diff().fillna(daily_series - daily_series.iloc[0])
    return daily_pnl


def compute_summary_stats(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float,
    daily_turnover: Dict[pd.Timestamp, float],
    periods_per_year: int = 252 * 1,
) -> Dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        sharpe = 0.0
        sortino = 0.0
        volatility = 0.0
    else:
        volatility = returns.std() * np.sqrt(periods_per_year)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        downside = returns[returns < 0]
        sortino = (
            (returns.mean() / downside.std()) * np.sqrt(periods_per_year)
            if not downside.empty and downside.std() > 0
            else 0.0
        )
    max_dd = compute_max_drawdown(equity_curve)
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1.0
    hit_rate = (
        float((trade_log["pnl"] > 0).mean()) if not trade_log.empty else 0.0
    )
    num_trades = len(trade_log)
    avg_daily_turnover = 0.0
    if daily_turnover:
        turnover_values = np.array(list(daily_turnover.values()))
        avg_daily_turnover = float(np.mean(turnover_values / initial_capital))
    summary = {
        "total_return": total_return,
        "cagr": compute_cagr(equity_curve),
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "num_trades": num_trades,
        "avg_daily_turnover": avg_daily_turnover,
    }
    return summary


def compute_cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = max((end - start).days / 365.25, 1e-9)
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min())
