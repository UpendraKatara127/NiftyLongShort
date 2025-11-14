import pandas as pd

from src.utils import config


def _cross_sectional_z(series):
    def transform(group):
        std = group.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=group.index)
        return (group - group.mean()) / std

    return series.groupby(level="datetime_hour").transform(transform)


def build_scores(pred_ret, pred_prob, alpha=config.ALPHA_STACK):
    z_ret = _cross_sectional_z(pred_ret)
    z_prob = _cross_sectional_z(pred_prob)
    return alpha * z_ret + (1 - alpha) * z_prob
