import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

from src.data_prep.load_data import load_day_data
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
        return df[["overnight_ret", "daily_range", "daily_vol", "volume_shock"]]
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


def build_feature_matrix(hourly):
    technical = compute_technical_features(hourly)
    statistical = build_statistical_features(hourly)
    regime = build_regime_features(hourly)
    daily = _build_daily_features(hourly)
    features = pd.concat([technical, statistical, regime, daily], axis=1).sort_index()
    return features


def build_dataset(hourly):
    features = build_feature_matrix(hourly)
    target = hourly["target"]
    label = hourly["label"]
    data = features.join(target.rename("target")).join(label.rename("label"))
    data = data.dropna(subset=["target"])
    return data
