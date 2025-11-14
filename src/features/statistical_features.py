import numpy as np
import pandas as pd

from src.utils import config


def _symbol_beta_features(df, window):
    df = df.copy()
    ret = df["ret"]
    idx_ret = df["index_ret"]
    ret_mean = ret.rolling(window, min_periods=window).mean()
    idx_mean = idx_ret.rolling(window, min_periods=window).mean()
    cov = (ret * idx_ret).rolling(window, min_periods=window).mean() - ret_mean * idx_mean
    var = (idx_ret.pow(2)).rolling(window, min_periods=window).mean() - idx_mean.pow(2)
    beta = cov / var.replace(0, np.nan)
    resid = ret - beta * idx_ret
    resid_vol = resid.rolling(window, min_periods=window).std()
    df["beta"] = beta
    df["resid_ret"] = resid
    df["resid_vol"] = resid_vol
    return df[["beta", "resid_ret", "resid_vol"]]


def _pca_features(returns_matrix, window):
    timestamps = returns_matrix.index
    symbols = returns_matrix.columns
    records = []
    global_records = []
    for idx in range(len(timestamps)):
        if idx + 1 < window:
            continue
        window_data = returns_matrix.iloc[idx - window + 1 : idx + 1]
        clean = window_data.fillna(0)
        cov = np.cov(clean.to_numpy().T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        total = eigvals.sum()
        probs = eigvals / total if total > 0 else np.zeros_like(eigvals)
        entropy = 0
        if total > 0:
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(eigvals))
        lambda1 = eigvals[0] if len(eigvals) else np.nan
        lambda_spread = total - lambda1 if total > 0 else np.nan
        ts = timestamps[idx]
        for sym_idx, symbol in enumerate(symbols):
            load1 = eigvecs[sym_idx, 0] if eigvecs.shape[1] > 0 else np.nan
            load2 = eigvecs[sym_idx, 1] if eigvecs.shape[1] > 1 else np.nan
            load3 = eigvecs[sym_idx, 2] if eigvecs.shape[1] > 2 else np.nan
            records.append(
                {
                    "datetime_hour": ts,
                    "symbol": symbol,
                    "pca_loading_1": load1,
                    "pca_loading_2": load2,
                    "pca_loading_3": load3,
                }
            )
        global_records.append(
            {
                "datetime_hour": ts,
                "lambda_1": lambda1,
                "lambda_spread": lambda_spread,
                "eigen_entropy": entropy,
            }
        )
    if records:
        loadings = pd.DataFrame.from_records(records).set_index(["datetime_hour", "symbol"])
    else:
        loadings = pd.DataFrame(columns=["pca_loading_1", "pca_loading_2", "pca_loading_3"])
        loadings.index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="datetime_hour"), pd.Index([], name="symbol")]
        )
    if global_records:
        global_df = pd.DataFrame.from_records(global_records).set_index("datetime_hour")
    else:
        global_df = pd.DataFrame(columns=["lambda_1", "lambda_spread", "eigen_entropy"])
        global_df.index = pd.DatetimeIndex([], name="datetime_hour")
    return loadings, global_df


def build_statistical_features(hourly):
    closes = hourly["close"]
    returns = closes.groupby(level="symbol").pct_change().rename("ret_1h")
    returns_frame = returns.to_frame("ret")
    ret_matrix = returns.unstack("symbol")
    index_ret = ret_matrix.mean(axis=1).rename("index_ret")
    returns_frame["index_ret"] = index_ret.reindex(returns_frame.index.get_level_values("datetime_hour")).values
    beta_block = (
        returns_frame.groupby(level="symbol", group_keys=False)
        .apply(_symbol_beta_features, config.BETA_WINDOW)
        .reorder_levels(["datetime_hour", "symbol"])
        .sort_index()
    )
    resid = beta_block["resid_ret"]
    ret_disp = returns.groupby(level="datetime_hour").std()
    resid_disp = resid.groupby(level="datetime_hour").std()
    volume = hourly["volume"]
    volume_disp = volume.groupby(level="datetime_hour").std()
    group_index = hourly.index.get_level_values("datetime_hour")
    stat = pd.DataFrame(index=hourly.index)
    stat["ret_dispersion"] = ret_disp.reindex(group_index).values
    stat["resid_dispersion"] = resid_disp.reindex(group_index).values
    stat["volume_dispersion"] = volume_disp.reindex(group_index).values
    volume_ratio = volume.groupby(level="symbol").transform(
        lambda s: s / s.rolling(20, min_periods=5).mean()
    )
    stat["volume_ratio"] = volume_ratio
    volume_ratio_disp = volume_ratio.groupby(level="datetime_hour").std()
    stat["volume_ratio_dispersion"] = volume_ratio_disp.reindex(group_index).values
    stat["beta"] = beta_block["beta"]
    stat["resid_ret"] = beta_block["resid_ret"]
    stat["resid_vol"] = beta_block["resid_vol"]
    pca_loadings, pca_global = _pca_features(ret_matrix, config.PCA_WINDOW)
    stat = stat.join(pca_loadings, how="left")
    stat["lambda_1"] = pca_global["lambda_1"].reindex(group_index).values
    stat["lambda_spread"] = pca_global["lambda_spread"].reindex(group_index).values
    stat["eigen_entropy"] = pca_global["eigen_entropy"].reindex(group_index).values
    vol = index_ret.rolling(config.VOL_LOOKBACK, min_periods=config.VOL_LOOKBACK).std()
    stat["index_vol"] = vol.reindex(group_index).values
    delta_vol = vol.diff()
    stat["delta_vol"] = delta_vol.reindex(group_index).values
    vol_of_vol = vol.rolling(config.VOL_OF_VOL_LOOKBACK, min_periods=config.VOL_OF_VOL_LOOKBACK).std()
    stat["vol_of_vol"] = vol_of_vol.reindex(group_index).values
    cross_z = stat["resid_ret"].groupby(level="datetime_hour").transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    stat["resid_cross_z"] = cross_z
    stat.columns = [f"stat_{c}" for c in stat.columns]
    return stat
