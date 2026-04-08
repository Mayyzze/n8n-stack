from typing import Literal, Optional

import numpy as np
import pandas as pd

_DATE_FMT = '%Y-%m-%d : %Hh%Mm%Ss'

# Maps supported window labels to their equivalent number of calendar days.
_WINDOW_DAYS: dict[str, int] = {'1d': 1, '5d': 5, '7d': 7, '1mo': 30, '3mo': 90, '1y': 365}

# In-process caches — keyed on id(data) so they are scoped to one DataFrame instance.
# Safe within a single script execution where data never changes.
_last_price_cache:    dict[tuple, tuple[str, float]] = {}
_price_at_time_cache: dict[tuple, tuple[str, float]] = {}
_valid_price_cache:   dict[tuple, Optional[float]]   = {}

Window = Literal['1d', '5d', '7d', '1mo', '3mo', '1y']


def _to_paris_tz(ts) -> str:
    """Converts a timestamp to a Paris-timezone string. Assumes UTC if tz-naive."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    return t.tz_convert('Europe/Paris').strftime(_DATE_FMT)


def get_last_price(data: pd.DataFrame, ticker: str, precision: int = 1) -> tuple[str, float]:
    """
    Returns the most recent valid (non-NaN) closing price for a ticker.

    Returns:
        (date_str, price) where date_str is formatted in Paris timezone.

    Raises:
        ValueError: if no valid price exists in the series.
    """
    key = (id(data), ticker, precision)
    if key in _last_price_cache:
        return _last_price_cache[key]
    close_series = data['Close', ticker].dropna()
    if close_series.empty:
        raise ValueError(f"No valid price found for ticker {ticker}")
    result = _to_paris_tz(close_series.index[-1]), round(close_series.iloc[-1], precision)
    _last_price_cache[key] = result
    return result


def get_valid_price_at_idx(data: pd.DataFrame, ticker: str, idx: int) -> Optional[float]:
    """
    Returns the most recent non-NaN 'Close' price at or before idx.

    Walks backwards from idx until a valid value is found.
    Supports negative indices (Python-style). Returns None if no valid value exists.
    """
    key = (id(data), ticker, idx)
    if key in _valid_price_cache:
        return _valid_price_cache[key]
    series = data['Close', ticker]
    n = len(series)
    if n == 0:
        return None
    # Normalise negative index, then clamp to valid range.
    idx = max(0, min(n + idx if idx < 0 else idx, n - 1))
    for i in range(idx, -1, -1):
        val = series.iloc[i]
        if not np.isnan(val):
            _valid_price_cache[key] = float(val)
            return float(val)
    _valid_price_cache[key] = None
    return None


def get_price_at_given_time(
    data: pd.DataFrame,
    ticker: str,
    window: Window,
    precision: int = 1,
) -> tuple[str, float]:
    """
    Returns the closing price approximately `window` ago relative to the last data point.

    The target date is computed as: last_date - window_days. The nearest available
    index entry is used, then the most recent non-NaN value at or before it.

    Returns:
        (date_str, price) where date_str is formatted in Paris timezone.

    Raises:
        ValueError: if the window is unsupported, out of range, or no valid price is found.
    """
    key = (id(data), ticker, window, precision)
    if key in _price_at_time_cache:
        return _price_at_time_cache[key]

    if window not in _WINDOW_DAYS:
        raise ValueError(f"Unsupported window: {window!r}")

    target_date = data.index[-1] - pd.Timedelta(days=_WINDOW_DAYS[window])
    if target_date < data.index[0]:
        raise ValueError("Requested time is before available data range")

    # Locate the index entry closest to target_date, then resolve a valid price.
    # The idx lookup is window-specific and identical across all tickers — cached via get_valid_price_at_idx.
    idx = data.index.get_indexer(pd.Index([target_date]), method='nearest')[0]
    price = get_valid_price_at_idx(data, ticker, idx)
    if price is None or np.isnan(price):
        raise ValueError(f"No valid price around {target_date} for {ticker}")

    result = _to_paris_tz(data.index[idx]), round(float(price), precision)
    _price_at_time_cache[key] = result
    return result


def get_price_evolution(
    data: pd.DataFrame,
    ticker: str,
    window: Window,
    precision: int = 1,
) -> float:
    """
    Returns the percentage price change between `window` ago and now.

    Formula: (last - prev) / prev * 100, rounded to `precision` decimal places.
    """
    _, last_price = get_last_price(data, ticker, precision)
    _, prev_price = get_price_at_given_time(data, ticker, window, precision)
    return round((last_price - prev_price) / prev_price * 100, precision)
