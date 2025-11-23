from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


class MinuteBarDataLoader:
    def __init__(
        self,
        data_root: str,
        universe_csv: str,
        market_open: str = "09:15",
        market_close: str = "15:30",
        timezone: str = "Asia/Kolkata",
    ):
        self.data_root = Path(data_root)
        self.minute_dir = self.data_root / "minute_wise"
        self.universe_csv = Path(universe_csv)
        self.market_open = self._parse_time(market_open)
        self.market_close = self._parse_time(market_close)
        self.timezone = timezone
        self._universe: Optional[List[str]] = None
        self._data_cache: Optional[pd.DataFrame] = None

    def load_universe(self) -> List[str]:
        if self._universe is not None:
            return self._universe
        df = pd.read_csv(self.universe_csv)
        symbols = (
            df["Security Symbol"].astype(str).str.strip().dropna().drop_duplicates().tolist()
        )
        self._ensure_symbol_files(symbols)
        self._universe = symbols
        return symbols

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._data_cache is None:
            self._data_cache = self._load_all_symbols()
        data = self._data_cache
        if start_date:
            start_ts = self._coerce_timestamp(start_date)
            data = data[data.index.get_level_values("datetime") >= start_ts]
        if end_date:
            end_ts = self._coerce_timestamp(end_date)
            data = data[data.index.get_level_values("datetime") <= end_ts]
        return data.copy()

    def get_rebalance_data(
        self,
        rebalance_times: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        data = self.load_data(start_date=start_date, end_date=end_date)
        rebalance_times = {t for t in rebalance_times}
        mask = (
            data.index.get_level_values("datetime").strftime("%H:%M").isin(rebalance_times)
        )
        return data[mask]

    def get_history_slice(
        self,
        timestamp: pd.Timestamp,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        data = self.load_data(start_date=start_date, end_date=end_date)
        mask = data.index.get_level_values("datetime") <= timestamp
        return data[mask]

    def _load_all_symbols(self) -> pd.DataFrame:
        universe = self.load_universe()
        frames = []
        for symbol in universe:
            symbol_frame = self._load_symbol_frame(symbol)
            if symbol_frame.empty:
                continue
            frames.append(symbol_frame)
        if not frames:
            raise ValueError("No symbol data loaded. Check data directory.")
        data = pd.concat(frames).sort_index()
        return data

    def _load_symbol_frame(self, symbol: str) -> pd.DataFrame:
        filename = f"{symbol}_minute_data.csv"
        path = self.minute_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing minute file for {symbol}: {path}")
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"Expected 'date' column in {path}")
        df["datetime"] = pd.to_datetime(df["date"], utc=False)
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize(self.timezone)
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(self.timezone)
        df = df.sort_values("datetime")
        times = df["datetime"].dt.time
        mask = (times >= self.market_open) & (times <= self.market_close)
        df = df[mask]
        df["symbol"] = symbol
        df = df.set_index(["datetime", "symbol"])
        columns = ["open", "high", "low", "close", "volume"]
        df = df[columns]
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _ensure_symbol_files(self, symbols: Iterable[str]) -> None:
        missing = []
        for symbol in symbols:
            if not (self.minute_dir / f"{symbol}_minute_data.csv").exists():
                missing.append(symbol)
        if missing:
            raise FileNotFoundError(
                f"Missing minute data files for symbols: {', '.join(sorted(missing))}"
            )

    def _parse_time(self, hhmm: str) -> time:
        return datetime.strptime(hhmm, "%H:%M").time()

    def _coerce_timestamp(self, ts_like: str) -> pd.Timestamp:
        ts = pd.Timestamp(ts_like)
        if ts.tzinfo is None:
            ts = ts.tz_localize(self.timezone)
        else:
            ts = ts.tz_convert(self.timezone)
        return ts
