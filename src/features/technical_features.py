import numpy as np
import pandas as pd


def _rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _kama(price, er_len=10, fast=2, slow=30):
    er = np.full_like(price, np.nan, dtype=float)
    sc = np.full_like(price, np.nan, dtype=float)
    kama = np.full_like(price, np.nan, dtype=float)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    for idx in range(er_len, len(price)):
        change = abs(price[idx] - price[idx - er_len])
        volatility = np.sum(
            np.abs(price[idx - er_len + 1 : idx + 1] - price[idx - er_len : idx])
        )
        er[idx] = change / volatility if volatility != 0 else 0
        sc[idx] = (er[idx] * (fast_sc - slow_sc) + slow_sc) ** 2
        if idx == er_len:
            kama[idx] = price[idx]
        else:
            kama[idx] = kama[idx - 1] + sc[idx] * (price[idx] - kama[idx - 1])
    return kama


def compute_technical_features(hourly):
    def per_symbol(df):
        close = df["close"]
        volume = df["volume"]
        features = pd.DataFrame(index=df.index)
        features["ret_1h"] = close.pct_change()
        features["ret_3h"] = close.pct_change(3)
        features["ret_5h"] = close.pct_change(5)
        features["volatility_5h"] = features["ret_1h"].rolling(5, min_periods=5).std()
        features["volatility_10h"] = features["ret_1h"].rolling(10, min_periods=10).std()
        sma3 = close.rolling(3, min_periods=3).mean()
        sma5 = close.rolling(5, min_periods=5).mean()
        features["price_ma_ratio_3h"] = close / sma3
        features["price_ma_ratio_5h"] = close / sma5
        ema5 = close.ewm(span=5, adjust=False).mean()
        ema10 = close.ewm(span=10, adjust=False).mean()
        features["ema_5h"] = ema5 / close
        features["ema_10h"] = ema10 / close
        features["rsi_14"] = _rsi(close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features["macd"] = macd
        kama_values = _kama(close.values)
        features["kama"] = kama_values
        features["price_kama_ratio"] = close / kama_values
        features["volume_rate"] = volume.pct_change()
        vol_mean = volume.rolling(20, min_periods=20).mean()
        vol_std = volume.rolling(20, min_periods=20).std()
        features["volume_zscore_20h"] = (volume - vol_mean) / vol_std
        features["intraday_range"] = (df["high"] - df["low"]) / close
        return features

    grouped = hourly.groupby(level="symbol", group_keys=False)
    features = grouped.apply(per_symbol)
    features.columns = [f"tech_{c}" for c in features.columns]
    return features
