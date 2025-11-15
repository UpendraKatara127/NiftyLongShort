import json
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.data_prep.build_panel import build_hourly_panel
from src.features.make_dataset import FeaturePipeline, build_dataset
from src.models.model_defs import (
    build_classification_model,
    build_regression_model,
    build_ridge_model,
)
from src.models.stacking import build_scores
from src.utils import config, paths


def _load_hourly():
    if paths.HOURLY_PANEL_PATH.exists():
        return pd.read_parquet(paths.HOURLY_PANEL_PATH)
    return build_hourly_panel()


def _parse_date(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(config.IST)
    else:
        ts = ts.tz_convert(config.IST)
    return ts


def _shap_summary(model, data, path):
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception:
        return
    sample = data.sample(min(len(data), 2000), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    shap.summary_plot(values, sample, show=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _feature_importance(model, columns):
    if not hasattr(model, "feature_importances_"):
        return {}
    values = model.feature_importances_
    return dict(sorted(zip(columns, values), key=lambda x: abs(x[1]), reverse=True))


def train_walkforward():
    print("Preparing dataset for modeling...", flush=True)
    hourly = _load_hourly()
    dataset = build_dataset(hourly)
    print(f"Dataset ready with {len(dataset)} samples and {len(dataset.columns) - 2} features", flush=True)
    feature_cols = [c for c in dataset.columns if c not in {"target", "label"}]
    features = dataset[feature_cols]
    target = dataset["target"]
    labels = dataset["label"].astype(int)
    timestamps = features.index.get_level_values("datetime_hour")
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(config.IST)
    else:
        timestamps = timestamps.tz_convert(config.IST)
    sessions = timestamps.normalize()
    unique_days = pd.Index(sorted(sessions.unique()))
    start_date = _parse_date(config.TRAIN_START)
    end_date = _parse_date(config.TEST_END)
    lookback = pd.Timedelta(days=config.LOOKBACK_DAYS)
    step = config.WALKFORWARD_STEP_DAYS
    signal_frames = []
    preds_ret = []
    preds_prob = []
    true_ret = []
    true_label = []
    latest_pipeline = None
    feature_store = None
    total_iterations = len(unique_days) - 1
    completed = 0
    for idx in range(total_iterations):
        current_day = unique_days[idx]
        next_day = unique_days[idx + 1]
        if current_day < start_date.normalize():
            continue
        if next_day > end_date.normalize():
            break
        if idx % step != 0:
            continue
        completed += 1
        if completed % 25 == 0 or completed == 1:
            print(f"Processing walk-forward split {completed}", flush=True)
        window_start = current_day - lookback
        train_mask = (sessions >= window_start) & (sessions <= current_day)
        test_mask = sessions == next_day
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_train = features.loc[train_mask]
        y_train = target.loc[train_mask]
        y_class = labels.loc[train_mask]
        if X_train.empty or y_train.empty:
            continue
        X_test = features.loc[test_mask]
        y_test = target.loc[test_mask]
        y_label_test = labels.loc[test_mask]
        pipeline = FeaturePipeline()
        print(
            f"  Window {window_start.date()} -> {current_day.date()} train={len(X_train)} rows, test={len(X_test)} rows",
            flush=True,
        )
        t0 = time.perf_counter()
        X_train_proc = pipeline.fit_transform(X_train)
        X_test_proc = pipeline.transform(X_test)
        print(f"  Features transformed in {time.perf_counter() - t0:.2f}s", flush=True)
        reg_model = build_regression_model()
        ridge_model = build_ridge_model()
        clf_model = build_classification_model()
        t1 = time.perf_counter()
        reg_model.fit(X_train_proc, y_train)
        ridge_model.fit(X_train_proc, y_train)
        clf_model.fit(X_train_proc, y_class)
        print(f"  Models fit in {time.perf_counter() - t1:.2f}s", flush=True)
        pred_ret_xgb = pd.Series(reg_model.predict(X_test_proc), index=X_test.index)
        pred_ret_ridge = pd.Series(ridge_model.predict(X_test_proc), index=X_test.index)
        pred_ret = (pred_ret_xgb + pred_ret_ridge) / 2
        pred_prob = pd.Series(clf_model.predict_proba(X_test_proc)[:, 1], index=X_test.index)
        score = build_scores(pred_ret, pred_prob)
        frame = pd.DataFrame(
            {
                "pred_ret": pred_ret,
                "pred_ret_xgb": pred_ret_xgb,
                "pred_ret_ridge": pred_ret_ridge,
                "pred_prob": pred_prob,
                "score": score,
                "target": y_test,
                "label": y_label_test,
            }
        )
        signal_frames.append(frame)
        preds_ret.append(pred_ret)
        preds_prob.append(pred_prob)
        true_ret.append(y_test)
        true_label.append(y_label_test)
        model_stamp = current_day.strftime("%Y%m%d")
        paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dump(reg_model, paths.MODELS_DIR / f"regressor_{model_stamp}.pkl")
        dump(ridge_model, paths.MODELS_DIR / f"ridge_{model_stamp}.pkl")
        dump(clf_model, paths.MODELS_DIR / f"classifier_{model_stamp}.pkl")
        pipeline.save(paths.FEATURE_PIPELINE_DIR / f"pipeline_{model_stamp}.pkl")
        latest_pipeline = pipeline
        feature_store = X_train_proc
    if not signal_frames:
        raise RuntimeError("No walk-forward splits were produced")
    print("Aggregating signals across splits...")
    signals = pd.concat(signal_frames).sort_index()
    sig_idx = signals.index.get_level_values("datetime_hour")
    if sig_idx.tz is None:
        sig_idx = sig_idx.tz_localize(config.IST)
    else:
        sig_idx = sig_idx.tz_convert(config.IST)
    start_local = start_date if start_date.tzinfo else start_date.tz_localize(config.IST)
    end_local = end_date if end_date.tzinfo else end_date.tz_localize(config.IST)
    mask = (sig_idx >= start_local) & (sig_idx <= end_local)
    signals = signals.loc[mask]
    signals.to_parquet(paths.SIGNALS_PATH)
    if latest_pipeline is not None:
        latest_pipeline.save(paths.FEATURE_PIPELINE_DIR / "pipeline_latest.pkl")
    preds_ret_series = pd.concat(preds_ret)
    preds_prob_series = pd.concat(preds_prob)
    true_ret_series = pd.concat(true_ret)
    true_label_series = pd.concat(true_label)
    ic_per_bar = signals.groupby(level="datetime_hour").apply(
        lambda df: df["pred_ret"].corr(df["target"], method="spearman")
    )
    ic_value = ic_per_bar.dropna().mean()
    mse = mean_squared_error(true_ret_series, preds_ret_series)
    rmse = np.sqrt(mse)
    auc = roc_auc_score(true_label_series, preds_prob_series) if len(true_label_series.unique()) > 1 else np.nan
    metrics = {
        "information_coefficient": ic_value,
        "auc": auc,
        "mse": mse,
        "rmse": rmse,
    }
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.MODELS_DIR / "ml_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    latest_models = sorted(paths.MODELS_DIR.glob("regressor_*.pkl"))
    if latest_models:
        latest_reg = latest_models[-1]
        latest_clf = sorted(paths.MODELS_DIR.glob("classifier_*.pkl"))[-1]
        reg_model = load(latest_reg)
        clf_model = load(latest_clf)
        fi_reg = {k: float(v) for k, v in _feature_importance(reg_model, feature_cols).items()}
        fi_clf = {k: float(v) for k, v in _feature_importance(clf_model, feature_cols).items()}
        with open(paths.MODELS_DIR / "feature_importance.json", "w", encoding="utf-8") as fp:
            json.dump({"regression": fi_reg, "classification": fi_clf}, fp, indent=2)
        shap_path = paths.MODELS_DIR / "shap_summary.png"
        # Skip automatic SHAP inside training; use scripts/run_shap.py instead.
    return signals


if __name__ == "__main__":
    train_walkforward()
