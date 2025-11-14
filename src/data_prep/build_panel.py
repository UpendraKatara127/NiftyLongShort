import pandas as pd

from src.data_prep.load_data import load_minute_data
from src.data_prep.resample_hourly import build_hourly_bars
from src.utils import config, paths


def _compute_targets(hourly):
    def per_symbol(df):
        idx = df.index.get_level_values("datetime_hour")
        session = pd.Series(idx.normalize(), index=df.index)
        same_day = session == session.shift(-1)
        next_close = df["close"].shift(-1)
        ret = (next_close - df["close"]) / df["close"]
        df["target"] = ret.where(same_day)
        return df

    enriched = hourly.groupby(level="symbol", group_keys=False).apply(per_symbol)
    mask = enriched["target"].notna()
    labels = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
    labels.loc[mask] = (enriched.loc[mask, "target"] > 0).astype(int)
    enriched["label"] = labels
    return enriched


def build_hourly_panel(symbols=None, start=None, end=None, save=True):
    minute_df = load_minute_data(symbols=symbols, start=start, end=end)
    hourly = build_hourly_bars(minute_df)
    panel = _compute_targets(hourly)
    if save:
        paths.HOURLY_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(paths.HOURLY_PANEL_PATH)
    return panel
