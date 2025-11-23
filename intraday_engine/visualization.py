from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import compute_max_drawdown


def plot_equity_curve(equity_curve: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    equity_curve.plot(ax=ax, title="Equity Curve")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_drawdown(equity_curve: pd.Series):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    drawdown.plot(ax=ax, title="Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=drawdown.min() * 1.1)
    ax.annotate(
        f"Max DD: {compute_max_drawdown(equity_curve):.2%}",
        xy=(1, drawdown.min()),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
    )
    plt.tight_layout()
    return ax


def plot_daily_pnl(daily_pnl: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 3))
    daily_pnl.plot.bar(ax=ax, title="Daily P&L")
    ax.set_ylabel("P&L")
    ax.grid(True, alpha=0.3)

    if isinstance(daily_pnl.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # Use an automatic locator/formatter combo so the axis stays readable.
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
    else:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return ax


def plot_hourly_pnl_hist(hourly_pnl: pd.Series, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(hourly_pnl.dropna(), bins=bins, alpha=0.7)
    ax.set_title("Hourly P&L Distribution")
    ax.set_xlabel("P&L")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return ax
