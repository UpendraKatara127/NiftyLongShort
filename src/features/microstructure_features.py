import numpy as np
import pandas as pd

from src.data_prep.load_data import load_minute_data
from src.utils import config


def _interval_minutes(time_obj):
    return time_obj.hour * 60 + time_obj.minute


def _assign_hour_labels(timestamps):
    local = timestamps.dt.tz_convert(config.IST)
    session = local.dt.floor("D")
    minutes = local.dt.hour * 60 + local.dt.minute
    labels = np.full(len(timestamps), -1, dtype=int)
    for start, end, label in config.HOURLY_INTERVALS:
        start_min = _interval_minutes(start)
        end_min = _interval_minutes(end)
        label_min = _interval_minutes(label)
        mask = (minutes >= start_min) & (minutes < end_min)
        labels[mask] = label_min
    return session, labels


def _aggregate_group(group):
    group = group.sort_values("datetime")
    prices = group["close"].astype(float)
    volumes = group["volume"].astype(float)
    total_vol = volumes.sum()
    if total_vol > 0:
        vwap = (prices * volumes).sum() / total_vol
    else:
        vwap = np.nan
    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]
    drift = (last_price - vwap) / vwap if vwap and not np.isnan(vwap) else 0.0
    momentum = (last_price - first_price) / first_price if first_price != 0 else 0.0
    head_count = min(len(group), 5)
    tail_count = min(len(group), 5)
    head_vol = volumes.head(head_count).sum()
    tail_vol = volumes.tail(tail_count).sum()
    denom = head_vol + tail_vol
    imbalance = (head_vol - tail_vol) / denom if denom > 0 else 0.0
    high = group["high"].max()
    low = group["low"].min()
    range_ratio = (high - low) / first_price if first_price != 0 else 0.0
    intraday_returns = prices.pct_change().dropna()
    return pd.Series(
        {
            "micro_vwap_drift": drift,
            "micro_momentum": momentum,
            "micro_volume_imbalance": imbalance,
            "micro_range_ratio": range_ratio,
            "micro_return_vol": intraday_returns.std(),
        }
    )


def build_microstructure_features(hourly):
    timestamps = hourly.index.get_level_values("datetime_hour")
    start = timestamps.min()
    end = timestamps.max()
    minute_df = load_minute_data(start=start, end=end)
    if minute_df.empty:
        return pd.DataFrame(index=hourly.index)
    minute_df = minute_df.reset_index()
    session, labels = _assign_hour_labels(minute_df["datetime"])
    minute_df["session"] = session.values
    minute_df["label_minute"] = labels
    minute_df = minute_df[minute_df["label_minute"] >= 0]
    if minute_df.empty:
        return pd.DataFrame(index=hourly.index)
    minute_df["datetime_hour"] = minute_df["session"] + pd.to_timedelta(
        minute_df["label_minute"], unit="m"
    )
    grouped = minute_df.groupby(["symbol", "datetime_hour"])
    features = grouped.apply(_aggregate_group)
    features.index = features.index.set_names(["symbol", "datetime_hour"])
    features = features.reorder_levels(["datetime_hour", "symbol"]).sort_index()
    return features
