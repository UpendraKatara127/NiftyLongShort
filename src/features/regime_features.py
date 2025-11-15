import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.utils import config


def _compute_regime_probs(index_ret, window):
    probs = pd.DataFrame(
        index=index_ret.index, columns=["p_momentum", "p_meanreversion", "p_noise"]
    )
    for idx in range(window, len(index_ret)):
        window_vals = index_ret.iloc[idx - window : idx + 1].dropna()
        if len(window_vals) < window // 2:
            continue
        gm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
        gm.fit(window_vals.to_numpy().reshape(-1, 1))
        current = index_ret.iloc[idx]
        resp = gm.predict_proba([[current]])[0]
        means = gm.means_.ravel()
        abs_means = np.abs(means)
        noise_idx = abs_means.argmin()
        remaining = [i for i in range(3) if i != noise_idx]
        momentum_idx = remaining[0]
        meanrev_idx = remaining[1] if len(remaining) > 1 else remaining[0]
        if means[momentum_idx] < means[meanrev_idx]:
            momentum_idx, meanrev_idx = meanrev_idx, momentum_idx
        probs.iloc[idx] = [
            resp[momentum_idx],
            resp[meanrev_idx],
            resp[noise_idx],
        ]
    return probs


def build_regime_features(hourly):
    closes = hourly["close"]
    returns = closes.groupby(level="symbol").pct_change()
    ret_matrix = returns.unstack("symbol")
    index_ret = ret_matrix.mean(axis=1).fillna(0)
    probs = _compute_regime_probs(index_ret, config.REGIME_WINDOW)
    vol = index_ret.rolling(config.VOL_LOOKBACK, min_periods=config.VOL_LOOKBACK).std()
    vol_of_vol = vol.rolling(config.VOL_OF_VOL_LOOKBACK, min_periods=config.VOL_OF_VOL_LOOKBACK).std()
    features = pd.DataFrame(index=hourly.index)
    index_align = hourly.index.get_level_values("datetime_hour")
    features["index_return"] = index_ret.reindex(index_align).values
    features["p_momentum"] = probs["p_momentum"].reindex(index_align).values
    features["p_meanreversion"] = probs["p_meanreversion"].reindex(index_align).values
    features["p_noise"] = probs["p_noise"].reindex(index_align).values
    features["vol_of_vol"] = vol_of_vol.reindex(index_align).values
    features.columns = [f"regime_{c}" for c in features.columns]
    return features
