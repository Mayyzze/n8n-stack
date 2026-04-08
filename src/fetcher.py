import yfinance as yf
import hashlib
import os
import pickle
import time
import pandas as pd


def load_tickers(tickers: list, interval: str = '1h', period: str = '1y', cache_duration: int = 3600) -> pd.DataFrame|None:
    """
    Charge les données Yahoo Finance pour les tickers, avec cache local (pickle).
    """
    cache_dir = "/tmp/yfinance_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5((str(tickers) + interval + period).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    cache_time_path = os.path.join(cache_dir, f"{cache_key}.time")

    if os.path.exists(cache_path) and os.path.exists(cache_time_path):
        with open(cache_time_path, "r") as f:
            cache_time = float(f.read())
        if time.time() - cache_time < cache_duration:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    backoff = 1.0
    max_retries = 3
    for _ in range(1, max_retries + 1):
        try:
            data:pd.DataFrame|None = yf.download(tickers, interval=interval, period=period, auto_adjust=True, progress=False, threads=False)
            if data is None or data.empty:
                raise ValueError("yfinance returned empty data")
            if ('Close' not in data.columns) and not any(isinstance(c, tuple) and c[0] == 'Close' for c in data.columns):
                raise ValueError("Downloaded data does not contain 'Close' column")
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            with open(cache_time_path, "w") as f:
                f.write(str(time.time()))
            return data
        except Exception:
            time.sleep(backoff)
            backoff *= 2

    return data
