import pandas as pd

from src.utils import config, paths


def load_constituents():
    df = pd.read_csv(paths.CONSTITUENTS_PATH)
    df["Security Symbol"] = df["Security Symbol"].str.strip().str.upper()
    return df.drop_duplicates("Security Symbol")


def load_sector_mapping():
    df = pd.read_csv(paths.SECTOR_PATH)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    return df.set_index("symbol")["sector"].to_dict()


def _build_symbol_file_map(directory, suffix):
    mapping = {}
    for path in directory.glob("*.csv"):
        base = path.stem
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        mapping[base.upper()] = path
    return mapping


def _parse_timestamp(ts):
    if ts is None:
        return None
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        value = value.tz_localize(config.IST)
    else:
        value = value.tz_convert(config.IST)
    return value


def _filter_intraday(df):
    local = df["datetime"].dt.tz_convert(config.IST)
    mask = (local.dt.time >= config.TRADING_START) & (local.dt.time <= config.TRADING_END)
    return df[mask]


def load_minute_data(symbols=None, start=None, end=None):
    constituents = load_constituents()
    if symbols is None:
        symbols = constituents["Security Symbol"].tolist()
    minute_map = _build_symbol_file_map(paths.MINUTE_DATA_DIR, "_minute_data")
    start_ts = _parse_timestamp(start)
    end_ts = _parse_timestamp(end)
    frames = []
    for symbol in symbols:
        path = minute_map.get(symbol)
        if path is None:
            raise FileNotFoundError(f"Missing minute data for {symbol}")
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
        df = _filter_intraday(df)
        if start_ts is not None:
            df = df[df["datetime"] >= start_ts]
        if end_ts is not None:
            df = df[df["datetime"] <= end_ts]
        df["symbol"] = symbol
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        df = df.sort_values("datetime")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.set_index(["datetime", "symbol"]).sort_index()
    return data[["open", "high", "low", "close", "volume"]]


def load_day_data(symbols=None):
    constituents = load_constituents()
    if symbols is None:
        symbols = constituents["Security Symbol"].tolist()
    day_map = _build_symbol_file_map(paths.DAY_DATA_DIR, "_daywise_data")
    frames = []
    for symbol in symbols:
        path = day_map.get(symbol)
        if path is None:
            raise FileNotFoundError(f"Missing daily data for {symbol}")
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
        df["symbol"] = symbol
        df = df.sort_values("datetime")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.set_index(["datetime", "symbol"]).sort_index()
    return data[["open", "high", "low", "close", "volume"]]
