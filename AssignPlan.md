# 📘 NIFTY 50 Hourly ML Long–Short Strategy — Project Specification
Hybrid Statistical + Technical | Market-Neutral | backtesting.py | Files

---

## 1. Data Inputs

### 1.1 Constituents

`data/nifty_50_constituents.csv`  
Columns: `Security Symbol, Security Name`

Symbols are always read from this file; none are hard-coded.

### 1.2 Price Data

- `data/minute_wise/<SYMBOL>.csv` – 1-min OHLCV per symbol  
- `data/day_wise/<SYMBOL>.csv` – daily OHLCV (only for extensions)

---

## 2. Project Directory Structure

```text
project/
  data/
    minute_wise/
    day_wise/
    nifty_50_constituents.csv
  src/
    data_prep/
      load_data.py
      resample_hourly.py
      build_panel.py
    features/
      technical_features.py
      statistical_features.py
      regime_features.py
      make_dataset.py
    models/
      model_defs.py
      stacking.py
      train_walkforward.py
    backtest/
      portfolio_book.py
      ml_longshort_strategy.py
      run_backtest.py
    utils/
      config.py
      paths.py
  outputs/
    feature_pipelines/
    models/
    backtests/
  notebooks/
  README.md
```

All `.py` files will contain **no comments** and **no docstrings**.

---

## 3. Data Pipeline

### 3.1 Load Constituents

- Read `data/nifty_50_constituents.csv`.  
- Extract unique `Security Symbol` as the list of tickers.  
- Validate that each ticker has a corresponding file in `data/minute_wise/`.

### 3.2 Load Minute Data

For each symbol in the constituents list:

- Read `data/minute_wise/<SYMBOL>.csv`.  
- Parse `datetime` to timezone-aware IST.  
- Filter intraday times: `09:15–15:30`.  
- Add `symbol` column equal to the ticker.  

Concatenate all into a single DataFrame with index `(datetime, symbol)` and columns:  
`open, high, low, close, volume`.

### 3.3 Hourly Bar Construction

Rebalance timestamps (per trading day):

- `10:15`  
- `11:15`  
- `12:15`  
- `13:15`  
- `14:15`

From minute data, build custom hourly bars aligned to these labels:

- Interval 1: 09:15–10:15 → label 10:15  
- Interval 2: 10:15–11:15 → label 11:15  
- Interval 3: 11:15–12:15 → label 12:15  
- Interval 4: 12:15–13:15 → label 13:15  
- Interval 5: 13:15–14:15 → label 14:15  

Aggregation per symbol and interval:

- `open`: first  
- `high`: max  
- `low`: min  
- `close`: last  
- `volume`: sum  

Save as `outputs/hourly_panel.parquet` with MultiIndex `(datetime_hour, symbol)`.

### 3.4 Label Definition

For each `(symbol, t)` where `t` is an hourly bar with a following bar `t+1`:

```text
y = (close(t+1) - close(t)) / close(t)
```

Drop last bar of each day where next-hour close is not available.

---

## 4. Feature Engineering

Features are computed using only information up to time `t`.

### 4.1 Technical Features (Per Symbol, Hourly)

Per `(symbol, t)` on the hourly series:

- Returns:  
  - `ret_1h` = `close_t / close_{t-1} - 1`  
  - `ret_2h`  
  - `ret_4h`  
- Moving averages and trends:  
  - `ema_3`, `ema_6`, `ema_12` on close  
  - `ema_spread_3_12 = ema_3 - ema_12`  
- Volatility:  
  - rolling std of `ret_1h` over windows (e.g. 6, 12)  
  - ATR(6) using hourly OHLC  
- Candle structure:  
  - body ratio `(close - open) / max(high - low, eps)`  
  - upper and lower wick ratios  
  - gap vs previous close `(open_t / close_{t-1} - 1)`  
- Volume:  
  - `vol_ratio_5 = volume_t / mean(volume_{t-1..t-5})`  
  - volume z-score over rolling window  
- Intraday position:  
  - slot index from `{1,2,3,4,5}` (one-hot or integer)

### 4.2 Statistical Features (Cross-Sectional and Factor)

At each hour `t`, across all stocks:

#### Cross-Sectional Z-Scores

For selected features `f` (e.g. `ret_1h`, `vol_ratio_5`, residual return):

- `cs_z_f(i,t) = (f(i,t) - mean_i f(i,t)) / std_i f(i,t)`

#### Cross-Sectional Dispersions

Per hour `t`:

- dispersion of raw returns: `std_i(ret_1h(i,t))`  
- dispersion of residual returns (defined below)  
- dispersion of volume and volume ratios  

#### Residual Model vs Index

- Compute index hourly returns (e.g. equal-weight cross-section or external NIFTY index series).  
- For each stock, estimate rolling beta to index returns.  
- Residual return: `resid_ret = ret_1h_stock - beta * ret_1h_index`.  
- Residual features:  
  - `resid_ret`  
  - residual z-score cross-sectionally  
  - residual volatility over rolling window  

#### PCA Structure

- On a rolling window (e.g. last 100–200 hours), build a 50-dim vector of stock returns at each hour.  
- Run PCA on the window.  
- Per stock, record loading on first 3 principal components at time `t`.  
- Global PCA features at time `t`:  
  - first eigenvalue `λ1` (market mode strength)  
  - dispersion `Σλ - λ1`  
  - eigenvalue entropy (normalized eigenvalue distribution entropy)

#### Volatility Structure

From index or cross-sectional returns:

- realized volatility (rolling std of index returns)  
- `delta_vol = vol_t - vol_{t-1}`  
- vol-of-vol = rolling std of `vol_t`  

### 4.3 Regime Features

From index hourly returns:

- Fit a small HMM or Markov-switching AR(1).  
- At each `t`, compute regime probabilities:  
  - `p_momentum`  
  - `p_meanreversion`  
  - `p_noise`  

From realized volatility:

- Bucket vol into regimes: `low`, `medium`, `high`.  
- One-hot encode vol regimes.

All regime probabilities and regime indicators become features.

### 4.4 Feature Pipeline

- Winsorize or clip extreme values.  
- Fill missing values.  
- Standardize or normalize features.  
- Implement as a transform object and fit only on training data.  
- Save fitted pipeline to `outputs/feature_pipelines/pipeline.pkl`.

---

## 5. Modeling

### 5.1 Target

- Regression target: cross-sectional z-score of the next-hour log return across all NIFTY 50 constituents at each rebalance time.  
- Binary target for classification: `label = 1 if z(log_ret_1h) > 0 else 0`.

### 5.2 Base Models

Two base models use the same feature set:

- `M1`: regression model (e.g. LightGBM, XGBoost, or GradientBoostingRegressor) to predict `y`.  
- `M2`: classification model (e.g. LightGBM/XGBoost classifier or LogisticRegression) to predict `P(label = 1)`.

### 5.3 Stacked Score

For each `(symbol, t)` with predictions:

1. Compute cross-sectional z-scores of `pred_ret` and `pred_prob` across symbols at time `t`.  
2. Final stacked score:

```text
score = α * z(pred_ret) + (1 - α) * z(pred_prob)
```

with `α` around `0.5` (to be tuned).

This `score` is used for cross-sectional ranking.

### 5.4 Optional Regime-Aware Models (Extension)

Optional extension:

- Train separate models per volatility regime (`low`, `medium`, `high`).  
- At inference time, combine using regime probabilities:

```text
score = Σ_k p_regime_k * score_k
```

where each `score_k` is the stacked score from the model trained for regime `k`.

---

## 6. Walk-Forward Training

### 6.1 Time Periods

- In-sample training + main backtest: April 2021 – March 2024.  
- Final out-of-sample test: April 2024 – March 2025.

### 6.2 Rolling Procedure

Choose a lookback window of 3–6 months of hourly data.

For each retrain date `D` (daily or weekly step):

1. `train_start = D - lookback_period`  
2. `train_end = D`  
3. Use data between `train_start` and `train_end` to fit feature pipeline and models `M1`, `M2`.  
4. Generate predictions for the next trading day (`D+1`) on all five rebalance hours.  
5. Store predictions in a signals table.

The signals table:

```text
outputs/signals.parquet

index: (datetime_hour, symbol)
columns: [pred_ret, pred_prob, score, optional_regime_features...]
```

All predictions must be strictly out-of-sample with respect to their training windows.

---

## 7. Backtesting Using `backtesting.py`

### 7.1 Portfolio and Rebalancing Rules

- Total capital: 10,000,000.  
- Positions are rebalanced at each of the five hourly times.  
- Holding period: exactly one hour between rebalances.  
- Strategy is market-neutral:  
  - 100% long exposure  
  - 100% short exposure  
  - 200% gross exposure.

At each rebalance hour `t`:

1. Close all open positions.  
2. Read `score` for all available symbols at time `t` from `signals.parquet`.  
3. Rank symbols by `score` descending.  
4. Select:  
   - top 10% as long basket  
   - bottom 10% as short basket.  
5. Allocate equal weight to each stock within long and short sides.  
6. Long notional = 100% of capital; short notional = 100% of capital.  
7. Compute quantity per stock:

```text
notional_per_long = capital / N_long
shares_long = notional_per_long / price_t
```

Similarly for shorts.

8. Transaction cost: 10 bps (0.10%) per trade per side on notional, applied on both entry and exit.

### 7.2 Implementation Sketch

- Use `backtesting.Backtest` with a custom `ml_longshort_strategy.py` strategy class.  
- Provide the strategy with access to:  
  - price panel of all symbols  
  - signals table with scores.  
- Use a `PortfolioBook` helper:

  - Tracks positions per symbol.  
  - Applies fills at hourly close prices.  
  - Calculates P&L and transaction costs.  
  - Logs each trade.

### 7.3 Backtest Outputs

Write outputs to `outputs/backtests/`:

- `trade_log.csv` with columns:  
  - `timestamp`  
  - `symbol`  
  - `side`  
  - `entry_price`  
  - `exit_price`  
  - `quantity`  
  - `pnl`  
  - `pnl_pct`  

- Equity curve plot: `equity_curve.png`.  
- Performance summary: `performance.json` or `performance.md` with metrics.

---

## 8. Evaluation

### 8.1 ML-Level Metrics

Use only out-of-sample predictions from the walk-forward process:

- IC (Information Coefficient): Spearman rank correlation between predicted scores and realized next-hour returns, computed per hour and averaged.  
- AUC for the classification view (using `pred_prob` and binary label).  
- MSE / RMSE of regression predictions.  
- Feature importance from the tree models.  
- SHAP summary plots for interpretation.

### 8.2 Strategy-Level Metrics

From the backtest results:

- CAGR.  
- Sharpe ratio.  
- Sortino ratio.  
- Maximum drawdown.  
- Hit rate (fraction of profitable trades).  
- Average daily turnover (notional traded / capital).  
- Separate performance of long leg and short leg.  
- Average long-short spread return at each rebalance (mean long basket return minus mean short basket return).

---

## 9. Extensions (All Included)

### 9.1 Sector Neutrality

- Obtain sector classification for each symbol from an external mapping.  
- At each rebalance, adjust weights so that net exposure per sector is approximately zero on both long and short sides.

### 9.2 Volatility Scaling

- Use index volatility or vol-of-vol to scale gross exposure.  
- Lower target exposure during very high vol-of-vol regimes.  
- Example: multiply gross exposure by a factor in `[0.5, 1.0]` depending on volatility bucket.

### 9.3 Maximum Constraints

- Max weight per stock (e.g. 5–10% of each leg).  
- Max turnover limit per day; if exceeded, scale trades down.  
- Skip trading in hours where:  
  - cross-sectional dispersion is below a threshold, or  
  - λ1 (market mode) is extremely high, indicating index-dominated moves.

### 9.4 Daily Data Integration

Using `data/day_wise/<SYMBOL>.csv`:

- Compute overnight return.  
- Compute previous-day realized volatility and volume shock.  
- Lag these daily features into the hourly dataset (use previous trading day values).  
- Include them as additional predictors in the ML models.

### 9.5 Advanced Statistical Layers

Optional advanced work:

- Fit Markov-switching GARCH to index returns for richer volatility regimes.  
- Track transitions in PCA eigenstructure (e.g. jumps in λ1, entropy changes).  
- Use entropy-based filters on cross-sectional returns or volumes to identify noisy vs structured regimes.

---

## 10. Deliverables

Final deliverables for the project:

- `outputs/hourly_panel.parquet`  
- `outputs/feature_pipelines/pipeline.pkl`  
- `outputs/signals.parquet`  
- Saved model checkpoints in `outputs/models/`  
- `outputs/backtests/trade_log.csv`  
- Equity curve and performance report in `outputs/backtests/`  
- A final Markdown or PDF report summarizing:  
  - data and preprocessing  
  - features  
  - modeling approach  
  - ML metrics  
  - backtest results  
  - interpretation and limitations

