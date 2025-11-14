import pandas as pd

from src.utils import config


def _interval_minutes(value):
    return value.hour * 60 + value.minute


def _assign_hour_labels(index):
    local = index.tz_convert(config.IST)
    session = local.normalize()
    minutes = local.hour * 60 + local.minute
    label_minutes = pd.Series([-1] * len(index), index=index)
    for start, end, label in config.HOURLY_INTERVALS:
        start_minute = _interval_minutes(start)
        end_minute = _interval_minutes(end)
        label_minute = _interval_minutes(label)
        mask = (minutes >= start_minute) & (minutes < end_minute)
        label_minutes.loc[mask] = label_minute
    return session, label_minutes


def build_hourly_bars(minute_df):
    if minute_df.empty:
        return minute_df
    minute_df = minute_df.sort_index()
    minute_df = minute_df.reset_index()
    idx = pd.DatetimeIndex(minute_df["datetime"])
    if idx.tz is None:
        idx = idx.tz_localize(config.IST)
    session, label_minutes = _assign_hour_labels(idx)
    minute_df = minute_df.copy()
    minute_df["session"] = session.values
    minute_df["label_minute"] = label_minutes.values
    minute_df = minute_df[minute_df["label_minute"] >= 0]
    if minute_df.empty:
        return minute_df
    minute_df["datetime_hour"] = minute_df["session"] + pd.to_timedelta(minute_df["label_minute"], unit="m")
    grouped = (
        minute_df.groupby(["symbol", "datetime_hour"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )
    hourly = grouped.reorder_levels(["datetime_hour", "symbol"]).sort_index()
    hourly.index = hourly.index.set_names(["datetime_hour", "symbol"])
    return hourly
