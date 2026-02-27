import hashlib
import json
import logging
import os
import pickle
import time

import pandas as pd
import yfinance as yf
from curl_cffi import requests as _curl_requests
from portfolio import ASSET_TYPES, PORTFOLIO_DICT, START_DATE

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_YF_SESSION = _curl_requests.Session(impersonate="chrome")
_PARIS = "Europe/Paris"

# Tickers cotés en USD → conversion EUR requise
USD_TICKERS: frozenset[str] = frozenset({"BTC-USD", "GC=F", "XDW0L.XC", "HSTE.L", "CEMA.L", "TTE"})
DELTA_DAYS:  dict[str, int]  = {"1d": 1, "5d": 5, "7d": 7, "1mo": 30, "3mo": 90, "1y": 365}


# ─── Primitives ───────────────────────────────────────────────────────────────

def _load_tickers(tickers: list[str], interval: str = "1d", period: str = "2y", ttl: int = 3600) -> pd.DataFrame:
    """Télécharge les données Yahoo Finance avec cache pickle + session Chrome."""
    os.makedirs("cache", exist_ok=True)
    path = os.path.join("cache", hashlib.md5((str(tickers) + interval + period).encode()).hexdigest() + ".pkl")

    if os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl:
        with open(path, "rb") as f:
            return pickle.load(f)

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            logging.info(f"Downloading {len(tickers)} tickers (attempt {attempt})…")
            df = yf.download(tickers, interval=interval, period=period,
                             auto_adjust=True, progress=False, threads=False,
                             session=_YF_SESSION)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            with open(path, "wb") as f:
                pickle.dump(df, f)
            return df
        except Exception as e:
            last_exc = e
            logging.warning(f"Attempt {attempt} failed: {e}")
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Download failed after 3 attempts: {last_exc}")


def _price_at_idx(series: pd.Series, idx: int) -> float | None:
    """Prix non-NaN le plus récent jusqu'à idx (inclus), ou None si absent."""
    valid = series.iloc[:idx + 1].dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def _last_price(data: pd.DataFrame, ticker: str, precision: int = 2) -> tuple[str, float]:
    """Dernier prix valide d'un ticker avec sa date en heure de Paris."""
    series = data["Close", ticker].dropna()
    if series.empty:
        raise ValueError(f"No valid price for {ticker}")
    ts = series.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    date_str = ts.tz_convert(_PARIS).strftime("%Y-%m-%d %Hh%M")
    return date_str, round(float(series.iloc[-1]), precision)


def _price_at_window(data: pd.DataFrame, ticker: str, window: str, precision: int = 2) -> tuple[str, float]:
    """Prix à une fenêtre passée : '1d', '5d', '7d', '1mo', '3mo', '1y'."""
    if window not in DELTA_DAYS:
        raise ValueError(f"Unsupported window '{window}'")
    target = data.index[-1] - pd.Timedelta(days=DELTA_DAYS[window])
    if target < data.index[0]:
        raise ValueError(f"Window '{window}' predates available data")
    idx = data.index.get_indexer([target], method="nearest")[0]
    price = _price_at_idx(data["Close", ticker], idx)
    if price is None:
        raise ValueError(f"No valid price for {ticker} near {target.date()}")
    ts = data.index[idx]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(_PARIS).strftime("%Y-%m-%d %Hh%M"), round(price, precision)


def _evolution(data: pd.DataFrame, ticker: str, window: str, precision: int = 2) -> float:
    """Variation en % sur une fenêtre donnée."""
    _, now  = _last_price(data, ticker, precision)
    _, prev = _price_at_window(data, ticker, window, precision)
    return round((now - prev) / prev * 100, precision)


def _to_eur(price: float, ticker: str, eurusd: float) -> float:
    """Convertit un prix en EUR si le ticker est coté en USD."""
    return price / eurusd if ticker in USD_TICKERS else price


# ─── Sections JSON ────────────────────────────────────────────────────────────

def get_asset_section(data: pd.DataFrame, ticker: str, precision: int = 2, conversion_rate: float = 1.0) -> dict:
    _, price = _last_price(data, ticker, precision)
    return {
        "last_price":         round(price * conversion_rate, precision),
        "change_1d_percent":  _evolution(data, ticker, "1d",  precision),
        "change_1mo_percent": _evolution(data, ticker, "1mo", precision),
    }


def get_portfolio_value_eur(data: pd.DataFrame, portfolio: dict[str, float]) -> dict:
    """Valeur totale du portefeuille en EUR avec ventilation par actif."""
    _, eurusd = _last_price(data, "EURUSD=X", 4)
    breakdown: dict[str, float] = {}
    total = 0.0
    for ticker, qty in portfolio.items():
        _, price = _last_price(data, ticker, 2)
        val = _to_eur(price, ticker, eurusd) * qty
        breakdown[ticker] = round(val, 2)
        total += val
    return {"total_value_eur": round(total, 2), "breakdown": breakdown}


def get_portfolio_allocation_by_type(data: pd.DataFrame, portfolio: dict[str, float]) -> dict:
    """Répartition du portefeuille par classe d'actifs en EUR et en %."""
    _, eurusd = _last_price(data, "EURUSD=X", 4)
    alloc: dict[str, float] = {}
    total = 0.0
    for ticker, qty in portfolio.items():
        atype = ASSET_TYPES.get(ticker, "other")
        _, price = _last_price(data, ticker, 2)
        val = _to_eur(price, ticker, eurusd) * qty
        alloc[atype] = alloc.get(atype, 0.0) + val
        total += val
    return {
        "allocation_eur":     {k: round(v, 2) for k, v in alloc.items()},
        "allocation_percent": {k: round(v / total * 100, 2) for k, v in alloc.items()},
        "total_value_eur":    round(total, 2),
    }


def get_portfolio_performance_drilldown(data: pd.DataFrame, portfolio: dict[str, float], start_date: str) -> dict:
    """Performance par classe d'actifs depuis start_date + retour annualisé."""
    _, eurusd_now = _last_price(data, "EURUSD=X", 4)
    start_ts  = pd.Timestamp(start_date)
    idx_start = data.index.get_indexer([start_ts], method="nearest")[0]
    eurusd_start = _price_at_idx(data["Close", "EURUSD=X"], idx_start) or eurusd_now

    type_now:   dict[str, float] = {}
    type_start: dict[str, float] = {}

    for ticker, qty in portfolio.items():
        try:
            _, price_now = _last_price(data, ticker, 2)
            price_start  = _price_at_idx(data["Close", ticker], idx_start)
        except Exception:
            logging.warning(f"Skipping {ticker}: no price data")
            continue
        if price_start is None:
            logging.warning(f"Skipping {ticker}: no data at start date")
            continue

        atype = ASSET_TYPES.get(ticker, "other")
        type_now[atype]   = type_now.get(atype, 0.0)   + _to_eur(price_now,   ticker, eurusd_now)   * qty
        type_start[atype] = type_start.get(atype, 0.0) + _to_eur(price_start, ticker, eurusd_start) * qty

    days = max(1, (data.index[-1] - start_ts).days)

    def _perf(now: float, start: float) -> dict:
        if start <= 0:
            return {"performance_since_start_percent": None, "annualized_return_percent": None}
        return {
            "performance_since_start_percent": round((now - start) / start * 100, 2),
            "annualized_return_percent":       round(((now / start) ** (365 / days) - 1) * 100, 2),
        }

    result = {atype: _perf(type_now[atype], type_start.get(atype, 0.0)) for atype in type_now}
    result["Global"] = _perf(sum(type_now.values()), sum(type_start.values()))
    return result


def get_macro_indicators(data: pd.DataFrame) -> dict:
    """Prix et évolutions du pétrole WTI (CL=F) et Brent (BZ=F)."""
    result: dict = {}
    for name, ticker in {"oil_wti_usd": "CL=F", "oil_brent_usd": "BZ=F"}.items():
        try:
            _, price = _last_price(data, ticker, 2)
            result[name] = {
                "price_usd":          price,
                "change_1d_percent":  _evolution(data, ticker, "1d",  2),
                "change_1mo_percent": _evolution(data, ticker, "1mo", 2),
                "change_1y_percent":  _evolution(data, ticker, "1y",  2),
            }
        except Exception as e:
            logging.warning(f"Macro {name}: {e}")
            result[name] = None
    return result


def get_market_gold_ratios(data: pd.DataFrame, eurusd: float, inrusd: float) -> dict:
    """
    Ratio marché / or pour 4 zones, normalisés en USD.
    Ratio croissant = le marché sur-performe l'or.
    """
    try:
        _, gold_now = _last_price(data, "GC=F", 2)
        gold_hist = {w: _price_at_window(data, "GC=F", w, 2)[1] for w in ("1d", "1mo", "1y")}
    except Exception as e:
        logging.warning(f"Gold unavailable for ratios: {e}")
        return {}

    # (ticker, facteur vers USD)
    markets = {
        "us_sp500_gold":    ("^GSPC",   1.0),
        "india_nifty_gold": ("^NSEI",   inrusd),
        "asia_em_gold":     ("CEMA.L",  1.0),
        "china_a_gold":     ("DBX9.DE", eurusd),
    }

    result: dict = {}
    for name, (ticker, fx) in markets.items():
        try:
            _, p_now = _last_price(data, ticker, 2)
            p_hist = {w: _price_at_window(data, ticker, w, 2)[1] for w in ("1d", "1mo", "1y")}
            r_now = round(p_now * fx / gold_now, 4)
            result[name] = {"ratio": r_now}
            for w in ("1d", "1mo", "1y"):
                r_hist = p_hist[w] * fx / gold_hist[w]
                result[name][f"change_{w}_percent"] = round((r_now - r_hist) / r_hist * 100, 2)
        except Exception as e:
            logging.warning(f"Ratio {name}: {e}")
            result[name] = None
    return result


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TICKERS = [
        "BTC-USD", "EURUSD=X", "INRUSD=X", "GC=F",
        "XDW0L.XC", "HSTE.L", "DBX9.DE", "CEMA.L", "TTE",
        "CL=F", "BZ=F",    # pétrole
        "^GSPC", "^NSEI",  # indices
    ]
    data = _load_tickers(TICKERS)
    _, eurusd = _last_price(data, "EURUSD=X", 4)
    _, inrusd = _last_price(data, "INRUSD=X", 6)

    print(json.dumps({
        "assets": {
            "BTC_USD":            get_asset_section(data, "BTC-USD"),
            "EUR_USD":            get_asset_section(data, "EURUSD=X", precision=4),
            "GOLD_USD":           get_asset_section(data, "GC=F"),
            "ENERGY_ETF_USD":     get_asset_section(data, "XDW0L.XC"),
            "TOTAL_ENERGIES_EUR": get_asset_section(data, "TTE"),
            "HKTech_USD":         get_asset_section(data, "HSTE.L"),
            "ChinaA_EUR":         get_asset_section(data, "DBX9.DE"),
            "AsiaEM_USD":         get_asset_section(data, "CEMA.L"),
        },
        "macro_indicators":    get_macro_indicators(data),
        "market_gold_ratios":  get_market_gold_ratios(data, eurusd, inrusd),
        "performance_by_type": get_portfolio_performance_drilldown(data, PORTFOLIO_DICT, START_DATE),
        "allocation_percent":  get_portfolio_allocation_by_type(data, PORTFOLIO_DICT)["allocation_percent"],
    }, indent=2))
