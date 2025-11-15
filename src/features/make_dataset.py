import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

from src.data_prep.load_data import load_day_data, load_sector_mapping
from src.features.microstructure_features import build_microstructure_features
from src.features.regime_features import build_regime_features
from src.features.statistical_features import build_statistical_features
from src.features.technical_features import compute_technical_features
from src.utils import config


class FeaturePipeline:
    def __init__(self, clip_bounds=config.PIPELINE_CLIP):
        self.clip_bounds = clip_bounds
        self.lower = None
        self.upper = None
        self.impute_values = None
        self.scaler = StandardScaler()

    def _coerce_numeric(self, df):
        bool_cols = df.select_dtypes(include=["bool"]).columns
        if len(bool_cols):
            df = df.copy()
            df[bool_cols] = df[bool_cols].astype(float)
        return df.apply(pd.to_numeric, errors="coerce")

    def fit(self, df):
        df = self._coerce_numeric(df)
        self.lower = df.quantile(self.clip_bounds[0])
        self.upper = df.quantile(self.clip_bounds[1])
        clipped = df.clip(lower=self.lower, upper=self.upper, axis=1)
        self.impute_values = clipped.median()
        filled = clipped.fillna(self.impute_values)
        self.scaler.fit(filled)
        return self

    def transform(self, df):
        df = self._coerce_numeric(df)
        clipped = df.clip(lower=self.lower, upper=self.upper, axis=1)
        filled = clipped.fillna(self.impute_values)
        values = self.scaler.transform(filled)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self, path)

    @staticmethod
    def load(path):
        return load(path)


def _build_daily_features(hourly):
    symbols = hourly.index.get_level_values("symbol").unique()
    daily = load_day_data(symbols=symbols)
    def per_symbol(df):
        df = df.copy()
        prev_close = df["close"].shift(1)
        df["overnight_ret"] = (df["open"] - prev_close) / prev_close
        df["daily_range"] = (df["high"] - df["low"]) / prev_close
        df["daily_vol"] = df["close"].pct_change().rolling(10, min_periods=5).std()
        df["volume_shock"] = df["volume"] / df["volume"].rolling(20, min_periods=5).mean()
        df["gap_flag"] = (df["overnight_ret"].abs() > 0.01).astype(int)
        df["volume_spike_flag"] = (df["volume_shock"] > 1.5).astype(int)
        return df[
            [
                "overnight_ret",
                "daily_range",
                "daily_vol",
                "volume_shock",
                "gap_flag",
                "volume_spike_flag",
            ]
        ]
    feats = daily.groupby(level="symbol", group_keys=False).apply(per_symbol)
    feats = feats.groupby(level="symbol").shift(1)
    session_index = feats.index.get_level_values("datetime").tz_convert(config.IST).normalize()
    feats.index = pd.MultiIndex.from_arrays(
        [session_index, feats.index.get_level_values("symbol")],
        names=["session", "symbol"],
    )
    hourly_idx = hourly.index.get_level_values("datetime_hour")
    if hourly_idx.tz is None:
        hourly_idx = hourly_idx.tz_localize(config.IST)
    else:
        hourly_idx = hourly_idx.tz_convert(config.IST)
    hourly_sessions = hourly_idx.normalize()
    align_index = pd.MultiIndex.from_arrays(
        [hourly_sessions, hourly.index.get_level_values("symbol")],
        names=["session", "symbol"],
    )
    aligned = feats.reindex(align_index)
    aligned.index = hourly.index
    aligned.columns = [f"daily_{c}" for c in aligned.columns]
    return aligned


def _add_cross_sectional_features(features):
    columns = [
        "stat_volume_ratio",
        "tech_volume_rate",
    ]

    def _zscore(series):
        std = series.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / std

    for col in columns:
        if col not in features.columns:
            continue
        z = features[col].groupby(level="datetime_hour").transform(_zscore)
        features[f"{col}_cs_z"] = z.fillna(0)
    return features


def _add_sector_relative_features(features):
    sector_map = load_sector_mapping()
    symbols = features.index.get_level_values("symbol")
    sectors = symbols.map(sector_map).fillna("UNKNOWN")
    datetimes = features.index.get_level_values("datetime_hour")
    columns = [
        "tech_ret_1h",
        "stat_resid_ret",
        "stat_volume_ratio",
        "tech_volume_rate",
    ]
    for col in columns:
        if col not in features.columns:
            continue
        temp = pd.DataFrame(
            {
                "value": features[col],
                "datetime": datetimes,
                "sector": sectors,
            },
            index=features.index,
        )
        sector_mean = temp.groupby(["datetime", "sector"])["value"].transform("mean")
        features[f"{col}_sector_rel"] = temp["value"] - sector_mean
    return features


def _cross_sectional_z(series):
    def transform(group):
        std = group.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return series.groupby(level="datetime_hour").transform(transform)


def build_feature_matrix(hourly):
    technical = compute_technical_features(hourly)
    statistical = build_statistical_features(hourly)
    regime = build_regime_features(hourly)
    daily = _build_daily_features(hourly)
    micro = build_microstructure_features(hourly)
    features = pd.concat([technical, statistical, regime, daily, micro], axis=1).sort_index()
    features = _add_cross_sectional_features(features)
    features = _add_sector_relative_features(features)
    return features


def build_dataset(hourly):
    features = build_feature_matrix(hourly)
    close = hourly["close"]
    future_close = close.groupby(level="symbol").shift(-1)
    log_ret = np.log(future_close / close)
    log_ret.name = "log_ret_1h"
    target = _cross_sectional_z(log_ret)
    target = target.where(log_ret.notna())
    label = (target > 0).astype(int)
    data = features.copy()
    data["log_ret_1h"] = log_ret
    data["target"] = target
    data["label"] = label
    data = data.dropna(subset=["target"])
    return data
