import numpy as np
import pandas as pd
from backtesting import Strategy

from src.backtest.portfolio_book import PortfolioBook
from src.data_prep.load_data import load_sector_mapping
from src.features.statistical_features import build_statistical_features
from src.utils import config


class MLLongShortStrategy(Strategy):
    hourly_panel = None
    signals = None
    ret_dispersion = None
    dispersion_threshold = None
    lambda_series = None
    lambda_threshold = None
    scale_factor = None
    sector_map = None

    @classmethod
    def configure(cls, hourly, signals):
        cls.hourly_panel = hourly
        cls.signals = signals
        cls.sector_map = load_sector_mapping()
        close_panel = hourly["close"].unstack("symbol")
        returns = close_panel.pct_change().dropna(how="all")
        cls.ret_dispersion = returns.std(axis=1)
        cls.dispersion_threshold = cls.ret_dispersion.quantile(0.2)
        stat_features = build_statistical_features(hourly)
        cls.lambda_series = stat_features["stat_lambda_1"].groupby(level="datetime_hour").first()
        cls.lambda_threshold = cls.lambda_series.quantile(0.9)
        cls.scale_factor = cls._build_volatility_scale(close_panel)

    @staticmethod
    def _build_volatility_scale(close_panel):
        index_ret = close_panel.pct_change().mean(axis=1).fillna(0)
        vol = index_ret.rolling(config.VOL_LOOKBACK, min_periods=config.VOL_LOOKBACK).std()
        vol_of_vol = vol.rolling(config.VOL_OF_VOL_LOOKBACK, min_periods=config.VOL_OF_VOL_LOOKBACK).std()
        low = vol_of_vol.quantile(0.05)
        high = vol_of_vol.quantile(0.95)
        span = high - low if high > low else 1
        normalized = (vol_of_vol - low) / span
        min_scale, max_scale = config.EXPOSURE_SCALE_RANGE
        scale = max_scale - normalized * (max_scale - min_scale)
        return scale.clip(lower=min_scale, upper=max_scale).fillna(max_scale)

    def init(self):
        self.book = PortfolioBook(config.CAPITAL, config.TRANSACTION_COST)
        self.daily_turnover = {}

    def next(self):
        timestamp = pd.Timestamp(self.data.index[-1])
        price_slice = self.hourly_panel.xs(timestamp, level="datetime_hour")
        prices = price_slice["close"].to_dict()
        self.book.close_positions(timestamp, prices)
        self.book.mark_nav(timestamp)
        scale = self.scale_factor.get(timestamp, config.EXPOSURE_SCALE_RANGE[1])
        orders = self._build_orders(timestamp, prices, scale)
        orders = self._apply_turnover_limit(timestamp, orders)
        if orders:
            self.book.open_positions(timestamp, orders)

    def _sector_adjust(self, weights):
        if not weights:
            return weights
        sector_totals = {}
        for symbol, weight in weights.items():
            sector = self.sector_map.get(symbol, "UNKNOWN")
            sector_totals.setdefault(sector, 0)
            sector_totals[sector] += weight
        if not sector_totals:
            return weights
        target = 1 / len(sector_totals)
        adjusted = {}
        for symbol, weight in weights.items():
            sector = self.sector_map.get(symbol, "UNKNOWN")
            total = sector_totals.get(sector, 1)
            factor = target / total if total else 0
            adjusted[symbol] = weight * factor
        total_weight = sum(adjusted.values())
        if total_weight == 0:
            return adjusted
        for symbol in adjusted:
            adjusted[symbol] /= total_weight
        return adjusted

    def _enforce_caps(self, weights):
        cap = config.MAX_WEIGHT_PER_STOCK
        if cap <= 0 or not weights:
            return weights
        weights = weights.copy()
        overflow = 0
        for symbol in list(weights.keys()):
            if weights[symbol] > cap:
                overflow += weights[symbol] - cap
                weights[symbol] = cap
        while overflow > 0:
            candidates = {s: w for s, w in weights.items() if w < cap}
            if not candidates:
                break
            total = sum(candidates.values())
            if total == 0:
                break
            for symbol, weight in candidates.items():
                room = cap - weight
                share = (weight / total) if total else 0
                add = min(room, overflow * share)
                weights[symbol] += add
                overflow -= add
                if overflow <= 1e-8:
                    break
        total_weight = sum(weights.values())
        if total_weight == 0:
            return weights
        for symbol in weights:
            weights[symbol] /= total_weight
        return weights

    def _build_orders(self, timestamp, prices, scale):
        try:
            slice_scores = self.signals.xs(timestamp, level="datetime_hour")
        except KeyError:
            return []
        if slice_scores.empty:
            return []
        dispersion = self.ret_dispersion.get(timestamp, np.nan)
        if pd.notna(dispersion) and dispersion < self.dispersion_threshold:
            return []
        lambda_value = self.lambda_series.get(timestamp, np.nan)
        if pd.notna(lambda_value) and lambda_value > self.lambda_threshold:
            return []
        count = max(int(len(slice_scores) * config.TOP_QUANTILE), 1)
        ranked = slice_scores.sort_values("score", ascending=False)
        longs = ranked.head(count)
        shorts = ranked.sort_values("score", ascending=True).head(count)
        shorts = shorts.loc[~shorts.index.isin(longs.index)]
        long_weights = {sym: 1 / len(longs) for sym in longs.index} if len(longs) else {}
        short_weights = {sym: 1 / len(shorts) for sym in shorts.index} if len(shorts) else {}
        long_weights = self._enforce_caps(self._sector_adjust(long_weights))
        short_weights = self._enforce_caps(self._sector_adjust(short_weights))
        orders = []
        long_notional = config.CAPITAL * config.LONG_EXPOSURE * scale
        short_notional = config.CAPITAL * config.SHORT_EXPOSURE * scale
        for symbol, weight in long_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            notional = long_notional * weight
            qty = notional / price
            orders.append({"symbol": symbol, "side": "long", "price": price, "quantity": qty})
        for symbol, weight in short_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            notional = short_notional * weight
            qty = notional / price
            orders.append({"symbol": symbol, "side": "short", "price": price, "quantity": qty})
        return orders

    def _apply_turnover_limit(self, timestamp, orders):
        if not orders:
            return orders
        day = timestamp.normalize()
        executed = self.daily_turnover.get(day, 0)
        limit = config.TURNOVER_LIMIT * config.CAPITAL
        total_notional = sum(order["price"] * abs(order["quantity"]) for order in orders)
        if executed >= limit:
            return []
        if executed + total_notional > limit and total_notional > 0:
            scale = max((limit - executed) / total_notional, 0)
            for order in orders:
                order["quantity"] *= scale
            total_notional *= scale
        self.daily_turnover[day] = executed + total_notional
        return [order for order in orders if order["quantity"] > 0]
