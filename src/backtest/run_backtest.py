import json
import os

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import Direction, SizeType

from src.data_prep.load_data import load_sector_mapping
from src.features.statistical_features import build_statistical_features
from src.utils import config, paths


def _load_hourly():
    if not paths.HOURLY_PANEL_PATH.exists():
        raise FileNotFoundError("Hourly panel not found. Run the data pipeline first.")
    return pd.read_parquet(paths.HOURLY_PANEL_PATH)


def _load_signals():
    if not paths.SIGNALS_PATH.exists():
        raise FileNotFoundError("Signals not found. Train models before backtesting.")
    return pd.read_parquet(paths.SIGNALS_PATH)


def _to_naive(index):
    if getattr(index, "tz", None) is None:
        return index
    return index.tz_convert(config.IST).tz_localize(None)


def _prepare_hourly(hourly):
    hourly = hourly.copy()
    dt = hourly.index.get_level_values("datetime_hour")
    symbols = hourly.index.get_level_values("symbol")
    hourly.index = pd.MultiIndex.from_arrays(
        [_to_naive(dt), symbols], names=hourly.index.names
    )
    return hourly


def _prepare_signals(signals):
    signals = signals.copy()
    dt = signals.index.get_level_values("datetime_hour")
    symbols = signals.index.get_level_values("symbol")
    signals.index = pd.MultiIndex.from_arrays(
        [_to_naive(dt), symbols], names=signals.index.names
    )
    return signals


def _build_vol_scale(close_panel):
    index_ret = close_panel.pct_change().mean(axis=1).fillna(0)
    vol = index_ret.rolling(config.VOL_LOOKBACK, min_periods=config.VOL_LOOKBACK).std()
    vol_of_vol = vol.rolling(config.VOL_OF_VOL_LOOKBACK, min_periods=config.VOL_OF_VOL_LOOKBACK).std()
    low = vol_of_vol.quantile(0.05)
    high = vol_of_vol.quantile(0.95)
    span = high - low if high > low else 1
    normalized = (vol_of_vol - low) / span
    min_scale, max_scale = config.EXPOSURE_SCALE_RANGE
    scale = max_scale - normalized * (max_scale - min_scale)
    return scale.clip(lower=min_scale, upper=max_scale).fillna(max_scale)


def _compute_filters(hourly):
    close_panel = hourly["close"].unstack("symbol").sort_index()
    returns = close_panel.pct_change().dropna(how="all")
    ret_dispersion = returns.std(axis=1)
    dispersion_threshold = ret_dispersion.quantile(0.2)
    stat_features = build_statistical_features(hourly)
    lambda_series = stat_features["stat_lambda_1"].groupby(level="datetime_hour").first()
    lambda_series.index = _to_naive(lambda_series.index)
    lambda_threshold = lambda_series.quantile(0.9)
    scale_factor = _build_vol_scale(close_panel)
    scale_factor.index = _to_naive(scale_factor.index)
    beta_table = (
        stat_features["stat_beta"].unstack("symbol").reindex(close_panel.index).sort_index()
    )
    return (
        close_panel,
        ret_dispersion,
        dispersion_threshold,
        lambda_series,
        lambda_threshold,
        scale_factor,
        beta_table,
    )


def _sector_adjust(weights, sector_map):
    if not weights:
        return weights
    sector_totals = {}
    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, "UNKNOWN")
        sector_totals.setdefault(sector, 0)
        sector_totals[sector] += weight
    if not sector_totals:
        return weights
    target = 1 / len(sector_totals)
    adjusted = {}
    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, "UNKNOWN")
        total = sector_totals.get(sector, 1)
        factor = target / total if total else 0
        adjusted[symbol] = weight * factor
    total_weight = sum(adjusted.values())
    if total_weight == 0:
        return adjusted
    for symbol in adjusted:
        adjusted[symbol] /= total_weight
    return adjusted


def _enforce_caps(weights):
    cap = config.MAX_WEIGHT_PER_STOCK
    if cap <= 0 or not weights:
        return weights
    weights = weights.copy()
    overflow = 0
    for symbol in list(weights.keys()):
        if weights[symbol] > cap:
            overflow += weights[symbol] - cap
            weights[symbol] = cap
    while overflow > 0:
        candidates = {s: w for s, w in weights.items() if w < cap}
        if not candidates:
            break
        total = sum(candidates.values())
        if total == 0:
            break
        for symbol, weight in candidates.items():
            room = cap - weight
            share = (weight / total) if total else 0
            add = min(room, overflow * share)
            weights[symbol] += add
            overflow -= add
            if overflow <= 1e-8:
                break
    total_weight = sum(weights.values())
    if total_weight == 0:
        return weights
    for symbol in weights:
        weights[symbol] /= total_weight
    return weights


def _build_weights(score_df, ret_dispersion, dispersion_threshold, lambda_series, lambda_threshold, scale_factor, beta_table, sector_map):
    weights = pd.DataFrame(0.0, index=score_df.index, columns=score_df.columns)
    daily_turnover = {}
    for ts in score_df.index:
        scores = score_df.loc[ts].dropna()
        if scores.empty:
            continue
        dispersion = ret_dispersion.get(ts, np.nan)
        if pd.notna(dispersion) and dispersion < dispersion_threshold:
            continue
        lambda_value = lambda_series.get(ts, np.nan)
        if pd.notna(lambda_value) and lambda_value > lambda_threshold:
            continue
        filtered = scores[abs(scores) >= config.MIN_SCORE_Z]
        if filtered.empty:
            continue
        count = max(int(len(scores) * config.TOP_QUANTILE), 1)
        count = min(count, config.MAX_BASKET_SIZE)
        long_candidates = filtered[filtered > 0]
        short_candidates = filtered[filtered < 0]
        longs = long_candidates.nlargest(count)
        shorts = short_candidates.nsmallest(count)
        if longs.empty and shorts.empty:
            continue
        long_weights = {sym: 1 / len(longs) for sym in longs.index} if len(longs) else {}
        short_weights = {sym: 1 / len(shorts) for sym in shorts.index} if len(shorts) else {}
        long_weights = _enforce_caps(_sector_adjust(long_weights, sector_map))
        short_weights = _enforce_caps(_sector_adjust(short_weights, sector_map))
        scale = scale_factor.get(ts, config.EXPOSURE_SCALE_RANGE[1])
        row = weights.loc[ts]
        if long_weights:
            long_total = config.LONG_EXPOSURE * scale
            for symbol, weight_value in long_weights.items():
                row[symbol] = long_total * weight_value
        if short_weights:
            short_total = config.SHORT_EXPOSURE * scale
            for symbol, weight_value in short_weights.items():
                row[symbol] = -short_total * weight_value
        beta_row = beta_table.loc[ts] if ts in beta_table.index else None
        if beta_row is not None:
            row = _beta_neutralize(row, beta_row, scale)
        notional = config.CAPITAL * row.abs().sum()
        day = pd.Timestamp(ts).normalize()
        limit = config.TURNOVER_LIMIT * config.CAPITAL
        executed = daily_turnover.get(day, 0)
        if executed >= limit or notional == 0:
            row[:] = 0
            continue
        if executed + notional > limit:
            scale_down = max((limit - executed) / notional, 0)
            row *= scale_down
            notional *= scale_down
        daily_turnover[day] = executed + notional
        weights.loc[ts] = row
    return weights.fillna(0.0), daily_turnover


def _beta_neutralize(row, beta_row, scale):
    active = row != 0
    if not active.any():
        return row
    betas = beta_row[active].fillna(0)
    denom = (betas**2).sum()
    if denom > 0:
        exposure = (row[active] * betas).sum()
        row.loc[active] = row.loc[active] - (exposure / denom) * betas
    pos_mask = row > 0
    neg_mask = row < 0
    pos_sum = row[pos_mask].sum()
    neg_sum = -row[neg_mask].sum()
    target_long = config.LONG_EXPOSURE * scale
    target_short = config.SHORT_EXPOSURE * scale
    if pos_sum > 0:
        row.loc[pos_mask] *= target_long / pos_sum
    if neg_sum > 0:
        row.loc[neg_mask] *= target_short / neg_sum
    return row


def _periods_per_year(signals):
    timestamps = pd.Index(sorted(signals.index.get_level_values("datetime_hour").unique()))
    per_day = pd.Series(1, index=timestamps).groupby(timestamps.normalize()).sum()
    return max(int(per_day.median()), 1) * 252


def _performance_metrics(trade_log, equity, signals, turnover):
    returns = equity["nav"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    periods = _periods_per_year(signals)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(periods) if std_ret > 0 else np.nan
    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = mean_ret / downside_std * np.sqrt(periods) if downside_std > 0 else np.nan
    start_nav = equity["nav"].iloc[0]
    end_nav = equity["nav"].iloc[-1]
    span_days = (equity["timestamp"].iloc[-1] - equity["timestamp"].iloc[0]).days
    years = max(span_days / 365.0, 1e-6)
    ratio = end_nav / start_nav if start_nav != 0 else np.nan
    cagr = ratio ** (1 / years) - 1 if ratio and ratio > 0 else np.nan
    cum_max = equity["nav"].cummax()
    drawdowns = equity["nav"] / cum_max - 1
    max_dd = drawdowns.min()
    hit_rate = (trade_log["pnl"] > 0).mean() if not trade_log.empty else np.nan
    long_pnl = trade_log.loc[trade_log["side"] == "long", "pnl"].sum()
    short_pnl = trade_log.loc[trade_log["side"] == "short", "pnl"].sum()
    spread_series = []
    if not trade_log.empty:
        grouped = trade_log.groupby("entry_time")
        for _, group in grouped:
            long_ret = group.loc[group["side"] == "long", "pnl_pct"].mean()
            short_ret = group.loc[group["side"] == "short", "pnl_pct"].mean()
            if pd.notna(long_ret) and pd.notna(short_ret):
                spread_series.append(long_ret - short_ret)
    spread = float(np.mean(spread_series)) if spread_series else np.nan
    turnover_values = list(turnover.values())
    avg_turnover = np.mean(turnover_values) / config.CAPITAL if turnover_values else 0
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "spread_return": spread,
        "avg_turnover": avg_turnover,
    }


def _build_vectorbt_portfolio(close_prices, weights):
    pf = vbt.Portfolio.from_orders(
        close=close_prices,
        price=close_prices,
        size=weights,
        size_type=SizeType.TargetPercent,
        direction=Direction.Both,
        cash_sharing=True,
        init_cash=config.CAPITAL,
        fees=config.TRANSACTION_COST,
        freq="H",
    )
    return pf


def _trade_log_from_portfolio(pf):
    records = pf.trades.records
    if records.size == 0:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "entry_price",
                "exit_price",
                "quantity",
                "pnl",
                "pnl_pct",
                "entry_time",
            ]
        )
    mask = records["exit_idx"] >= 0
    records = records[mask]
    if records.size == 0:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "entry_price",
                "exit_price",
                "quantity",
                "pnl",
                "pnl_pct",
                "entry_time",
            ]
        )
    close_index = pf.wrapper.index
    columns = pf.wrapper.columns
    side = np.where(records["direction"] == 1, "short", "long")
    trade_log = pd.DataFrame(
        {
            "timestamp": close_index[records["exit_idx"]],
            "symbol": columns[records["col"]],
            "side": side,
            "entry_price": records["entry_price"],
            "exit_price": records["exit_price"],
            "quantity": records["size"],
            "pnl": records["pnl"],
            "pnl_pct": records["return"],
            "entry_time": close_index[records["entry_idx"]],
        }
    )
    return trade_log.sort_values("timestamp").reset_index(drop=True)


def run_backtest():
    hourly = _prepare_hourly(_load_hourly())
    signals = _prepare_signals(_load_signals())
    close_panel, ret_dispersion, dispersion_threshold, lambda_series, lambda_threshold, scale_factor, beta_table = _compute_filters(hourly)
    sector_map = load_sector_mapping()
    score_df = signals["score"].unstack("symbol")
    common_index = score_df.index.intersection(close_panel.index)
    if common_index.empty:
        raise RuntimeError("Signals and price data have no overlapping timestamps.")
    closings = close_panel.reindex(common_index)
    score_df = score_df.reindex(common_index).reindex(columns=closings.columns)
    ret_dispersion = ret_dispersion.reindex(common_index)
    lambda_series = lambda_series.reindex(common_index)
    scale_factor = scale_factor.reindex(common_index)
    weights, turnover = _build_weights(
        score_df,
        ret_dispersion,
        dispersion_threshold,
        lambda_series,
        lambda_threshold,
        scale_factor,
        beta_table,
        sector_map,
    )
    weights = weights.reindex(closings.index).fillna(0.0)
    if weights.abs().sum().sum() == 0:
        raise RuntimeError("No valid weights were generated for the backtest.")
    pf = _build_vectorbt_portfolio(closings, weights)
    equity_series = pf.value()
    equity = pd.DataFrame(
        {"timestamp": equity_series.index, "nav": equity_series.values}
    )
    trade_log = _trade_log_from_portfolio(pf)
    if equity.empty:
        raise RuntimeError("Equity curve is empty, backtest failed.")
    paths.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(paths.TRADE_LOG_PATH, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(equity["timestamp"], equity["nav"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Net Asset Value")
    plt.tight_layout()
    plt.savefig(paths.EQUITY_CURVE_PATH)
    plt.close()
    metrics = _performance_metrics(trade_log, equity, signals, turnover)
    with open(paths.PERFORMANCE_PATH, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return trade_log, equity, metrics


if __name__ == "__main__":
    run_backtest()
