"""
Gavekal / Charles Gave macro indicators.

Framework: Four Quadrants (growth vs inflation axes) → regime detection → asset allocation.
Ref: "The General Theory of Portfolio Construction" - Charles Gave, Gavekal Research.
"""

import json, os, hashlib, pickle, time
import numpy as np
import pandas as pd
import yfinance as yf

# ── tickers ──────────────────────────────────────────────────────────────────
INDICATOR_TICKERS = [
    "^GSPC",      # S&P 500 (growth proxy)
    "CL=F",       # WTI crude (growth / commodity)
    "GC=F",       # Gold (inflation / safe-haven)
    "HG=F",       # Copper (global growth barometer)
    "TLT",        # 20Y Treasury ETF (deflation hedge)
    "^TNX",       # 10Y yield
    "^IRX",       # 13W yield (short-rate proxy)
    "DX-Y.NYB",   # USD index
]

MA_YEARS   = 7    # Gave's reference window for regime MAs
CACHE_DIR  = "cache"
CACHE_TTL  = 3600 * 6   # 6 h for macro data


# ── data loading ─────────────────────────────────────────────────────────────
def _load_indicator_data(period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Download indicator tickers with local pickle cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key   = hashlib.md5((str(INDICATOR_TICKERS) + period + interval).encode()).hexdigest()
    fdata = os.path.join(CACHE_DIR, f"gavekal_{key}.pkl")
    ftime = os.path.join(CACHE_DIR, f"gavekal_{key}.time")

    if os.path.exists(fdata) and os.path.exists(ftime):
        with open(ftime) as f:
            if time.time() - float(f.read()) < CACHE_TTL:
                with open(fdata, "rb") as g:
                    return pickle.load(g)

    backoff = 1.0
    for attempt in range(1, 4):
        try:
            data = yf.download(
                INDICATOR_TICKERS, period=period, interval=interval,
                auto_adjust=True, progress=False, threads=False
            )
            if data is None or data.empty:
                raise ValueError("empty data")
            with open(fdata, "wb") as f: pickle.dump(data, f)
            with open(ftime, "w") as f:  f.write(str(time.time()))
            return data
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Failed to download indicator data")


def _series(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract Close series for a ticker, drop NaNs."""
    try:
        return data["Close"][ticker].dropna()
    except (KeyError, TypeError):
        return data["Close", ticker].dropna()


# ── helpers ───────────────────────────────────────────────────────────────────
def _ratio_vs_ma(s: pd.Series, window: int) -> dict:
    """Return latest ratio value, its MA, and whether ratio is above MA."""
    ma  = s.rolling(window=window, min_periods=window // 2).mean()
    val = float(s.iloc[-1])
    mav = float(ma.iloc[-1])
    return {"value": round(val, 6), "ma": round(mav, 6), "above_ma": val >= mav}


# ── Four Quadrants ────────────────────────────────────────────────────────────
def get_four_quadrants(data: pd.DataFrame) -> dict:
    """
    Identify current Gavekal economic quadrant.

    X-axis (growth):    S&P 500 / WTI  vs 7-year MA
      above → boom,  below → bust

    Y-axis (inflation): Gold / TLT     vs 7-year MA
      above → inflationary,  below → deflationary

    Quadrant → preferred assets:
      Deflationary Boom  : equities, bonds, tech         → BUY dips
      Inflationary Boom  : gold, oil, commodities        → HOLD cautiously
      Deflationary Bust  : long Treasuries, cash         → REDUCE risk
      Stagflation        : gold, real assets, short EQ   → SELL rallies
    """
    spx    = _series(data, "^GSPC")
    wti    = _series(data, "CL=F")
    gold   = _series(data, "GC=F")
    tlt    = _series(data, "TLT")

    window = MA_YEARS * 252  # trading-day approximation

    df = pd.DataFrame({"spx": spx, "wti": wti, "gold": gold, "tlt": tlt}).dropna()

    growth_ratio = (df["spx"] / df["wti"]).rename("growth_ratio")
    infl_ratio   = (df["gold"] / df["tlt"]).rename("infl_ratio")

    gr = _ratio_vs_ma(growth_ratio, window)
    ir = _ratio_vs_ma(infl_ratio,   window)

    boom         = gr["above_ma"]
    inflationary = ir["above_ma"]

    if     boom and not inflationary: quadrant, signal, assets = "Deflationary Boom",  "BUY dips",       ["equities", "bonds", "tech", "HK_tech", "EM_Asia"]
    elif   boom and     inflationary: quadrant, signal, assets = "Inflationary Boom",   "HOLD cautiously",["gold", "energy", "commodities", "BTC"]
    elif not boom and not inflationary: quadrant, signal, assets = "Deflationary Bust","REDUCE risk",    ["long_treasuries", "cash", "defensive"]
    else:                              quadrant, signal, assets = "Stagflation",        "SELL rallies",   ["gold", "real_assets"]

    return {
        "quadrant":          quadrant,
        "tactical_signal":   signal,
        "recommended_assets": assets,
        "growth": {
            "ratio":    gr["value"],
            "ma":       gr["ma"],
            "signal":   "boom" if boom else "bust",
        },
        "inflation": {
            "ratio":    ir["value"],
            "ma":       ir["ma"],
            "signal":   "inflationary" if inflationary else "deflationary",
        },
    }


# ── Wicksellian spread ────────────────────────────────────────────────────────
def get_wicksellian_signal(data: pd.DataFrame) -> dict:
    """
    Wicksellian spread proxy.

    Gave's rule: if market rate < natural rate (real growth) → money too cheap
    → inflation or asset bubble ahead.

    Proxy:
      - Market rate   : 10Y nominal yield (^TNX)
      - Growth signal : Copper/Gold ratio normalised to a z-score (rising = growth)
      - Spread        : copper_gold_zscore - (10Y yield / its 3Y mean)

    Positive spread → stimulative (Wicksellian warning: inflation / bubble risk).
    Negative spread → restrictive (deflationary pressure).
    """
    tnx    = _series(data, "^TNX")
    copper = _series(data, "HG=F")
    gold   = _series(data, "GC=F")

    df = pd.DataFrame({"tnx": tnx, "copper": copper, "gold": gold}).dropna()

    # Normalised 10Y yield vs its 3-year mean
    y3  = 3 * 252
    tnx_ratio = df["tnx"] / df["tnx"].rolling(y3, min_periods=252).mean()

    # Copper/Gold z-score (252-day)
    cg      = df["copper"] / df["gold"]
    cg_mean = cg.rolling(252, min_periods=120).mean()
    cg_std  = cg.rolling(252, min_periods=120).std()
    cg_z    = (cg - cg_mean) / cg_std.replace(0, np.nan)

    spread = cg_z - tnx_ratio

    latest_spread = float(spread.iloc[-1])
    stimulative   = latest_spread > 0

    return {
        "wicksellian_spread":  round(latest_spread, 4),
        "regime":              "stimulative" if stimulative else "restrictive",
        "signal":              "inflation/bubble risk" if stimulative else "deflationary pressure",
        "tnx_vs_3y_mean":      round(float(tnx_ratio.iloc[-1]), 4),
        "copper_gold_zscore":  round(float(cg_z.iloc[-1]), 4),
    }


# ── Yield curve ───────────────────────────────────────────────────────────────
def get_yield_curve(data: pd.DataFrame) -> dict:
    """
    10Y - short-rate spread.
    Inversion → leading indicator of deflationary bust (Gave / standard).
    """
    tnx = _series(data, "^TNX")
    irx = _series(data, "^IRX")

    df  = pd.DataFrame({"tnx": tnx, "irx": irx}).dropna()
    df["spread"] = df["tnx"] - df["irx"]

    spread   = float(df["spread"].iloc[-1])
    inverted = spread < 0

    # steepening / flattening momentum (90-day change)
    momentum = float(df["spread"].iloc[-1] - df["spread"].iloc[-min(90, len(df)-1)])

    return {
        "yield_10y":         round(float(df["tnx"].iloc[-1]), 3),
        "yield_short":       round(float(df["irx"].iloc[-1]), 3),
        "spread_bp":         round(spread * 100, 1),
        "inverted":          inverted,
        "momentum_90d_bp":   round(momentum * 100, 1),
        "signal":            "recession_risk" if inverted else ("flattening" if momentum < 0 else "steepening"),
    }


# ── Copper / Gold ratio ───────────────────────────────────────────────────────
def get_copper_gold_ratio(data: pd.DataFrame) -> dict:
    """
    Copper/Gold ratio — Gave's real-time growth barometer.
    Rising → reflationary / growth optimism.
    Falling → deflationary / recession fear.
    """
    copper = _series(data, "HG=F")
    gold   = _series(data, "GC=F")

    df  = pd.DataFrame({"copper": copper, "gold": gold}).dropna()
    cg  = df["copper"] / df["gold"]

    ma90  = cg.rolling(90).mean()
    ma252 = cg.rolling(252).mean()

    val    = float(cg.iloc[-1])
    ma90v  = float(ma90.iloc[-1])
    ma252v = float(ma252.iloc[-1])

    trend_short = "rising" if val > ma90v  else "falling"
    trend_long  = "rising" if val > ma252v else "falling"

    return {
        "copper_gold_ratio": round(val,    6),
        "ma_90d":            round(ma90v,  6),
        "ma_252d":           round(ma252v, 6),
        "trend_short":       trend_short,
        "trend_long":        trend_long,
        "signal":            "reflationary" if (trend_short == "rising" and trend_long == "rising") else
                             "deflationary_pressure" if (trend_short == "falling" and trend_long == "falling") else
                             "mixed",
    }


# ── Dollar strength ───────────────────────────────────────────────────────────
def get_dollar_strength(data: pd.DataFrame) -> dict:
    """
    DXY trend.
    Strong dollar → EM / commodity headwind (Gave: bearish for non-US assets).
    Weak dollar   → EM / commodity tailwind.
    """
    dxy  = _series(data, "DX-Y.NYB")
    ma50 = dxy.rolling(50).mean()
    ma200= dxy.rolling(200).mean()

    val    = float(dxy.iloc[-1])
    ma50v  = float(ma50.iloc[-1])
    ma200v = float(ma200.iloc[-1])

    above_200 = val > ma200v
    momentum  = float(dxy.iloc[-1] - dxy.iloc[-min(90, len(dxy)-1)])

    return {
        "dxy":           round(val,    2),
        "ma_50d":        round(ma50v,  2),
        "ma_200d":       round(ma200v, 2),
        "above_ma200":   above_200,
        "momentum_90d":  round(momentum, 2),
        "regime":        "strong_dollar" if above_200 else "weak_dollar",
        "signal":        "EM/commodity headwind" if above_200 else "EM/commodity tailwind",
    }


# ── Portfolio quadrant fit ────────────────────────────────────────────────────
# Maps portfolio tickers to Gavekal quadrant favourability
_TICKER_QUADRANT_FIT = {
    "BTC-USD":  {"Deflationary Boom": "neutral",  "Inflationary Boom": "positive", "Deflationary Bust": "negative", "Stagflation": "positive"},
    "GC=F":     {"Deflationary Boom": "neutral",  "Inflationary Boom": "positive", "Deflationary Bust": "positive", "Stagflation": "positive"},
    "XDW0L.XC": {"Deflationary Boom": "neutral",  "Inflationary Boom": "positive", "Deflationary Bust": "negative", "Stagflation": "neutral"},
    "TTE":      {"Deflationary Boom": "neutral",  "Inflationary Boom": "positive", "Deflationary Bust": "negative", "Stagflation": "neutral"},
    "HSTE.L":   {"Deflationary Boom": "positive", "Inflationary Boom": "neutral",  "Deflationary Bust": "negative", "Stagflation": "negative"},
    "DBX9.DE":  {"Deflationary Boom": "positive", "Inflationary Boom": "neutral",  "Deflationary Bust": "negative", "Stagflation": "negative"},
    "CEMA.L":   {"Deflationary Boom": "positive", "Inflationary Boom": "neutral",  "Deflationary Bust": "negative", "Stagflation": "negative"},
}

def get_portfolio_quadrant_fit(quadrant: str) -> dict:
    """Score each portfolio position vs the current Gavekal quadrant."""
    return {
        ticker: fit.get(quadrant, "neutral")
        for ticker, fit in _TICKER_QUADRANT_FIT.items()
    }


# ── Master function ───────────────────────────────────────────────────────────
def get_all_gavekal_indicators() -> dict:
    """Fetch data once and return all Gavekal-inspired indicators."""
    data = _load_indicator_data(period="max", interval="1d")

    four_q = get_four_quadrants(data)

    return {
        "four_quadrants":        four_q,
        "portfolio_fit":         get_portfolio_quadrant_fit(four_q["quadrant"]),
        "wicksellian_signal":    get_wicksellian_signal(data),
        "yield_curve":           get_yield_curve(data),
        "copper_gold_ratio":     get_copper_gold_ratio(data),
        "dollar_strength":       get_dollar_strength(data),
    }


if __name__ == "__main__":
    result = get_all_gavekal_indicators()
    print(json.dumps(result, indent=2))
