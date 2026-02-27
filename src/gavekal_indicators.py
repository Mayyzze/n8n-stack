"""
Gavekal / Charles Gave macro indicators.

Framework: Four Quadrants (growth vs inflation axes) → regime detection → asset allocation.
Ref: "The General Theory of Portfolio Construction" - Charles Gave, Gavekal Research.

Axes:
  X (growth/energy efficiency): S&P 500 / WTI  vs 7-year SMA
  Y (currency quality/inflation): Gold / 10Y Treasury vs 7-year SMA

Quadrants:
  Disinflationary Boom  (right + bottom): best regime; buy dips
  Inflationary Boom     (right + top)   : real assets outperform
  Deflationary Bust     (left  + bottom): long Treasuries
  Stagflation           (left  + top)   : worst regime; sell rallies

Wicksellian spread (exact): BAA_real_yield - nominal_GDP_growth
  < 0        → stimulative / reflationary
  0-250 bps  → normal credit cycle
  > 250 bps  → recession imminent (predicted every recession since 1955)

Note: BAA yield and GDP growth are not available via yfinance.
A market-based proxy is used instead (copper/gold z-score vs normalised 10Y yield).
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
    "TLT",        # 20Y Treasury ETF (deflation hedge, kept for reference)
    "^TNX",       # 10Y yield
    "^IRX",       # 13W yield (short-rate proxy)
    "DX-Y.NYB",   # USD index
    "IEF",        # 7-10Y Treasury ETF (Gold/Treasury inflation ratio)
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
    spx  = _series(data, "^GSPC")
    wti  = _series(data, "CL=F")
    gold = _series(data, "GC=F")
    # IEF = 7-10Y Treasury ETF, best yfinance proxy for "10Y Treasury total return"
    ief  = _series(data, "IEF")

    window = MA_YEARS * 252  # trading-day approximation

    df = pd.DataFrame({"spx": spx, "wti": wti, "gold": gold, "ief": ief}).dropna()

    growth_ratio = (df["spx"] / df["wti"]).rename("growth_ratio")
    infl_ratio   = (df["gold"] / df["ief"]).rename("infl_ratio")

    gr = _ratio_vs_ma(growth_ratio, window)
    ir = _ratio_vs_ma(infl_ratio,   window)

    boom         = gr["above_ma"]
    inflationary = ir["above_ma"]

    if   boom and not inflationary: quadrant, signal, assets = "Deflationary Boom",  "BUY dips",        ["equities", "bonds", "tech", "HK_tech", "EM_Asia"]
    elif boom and     inflationary: quadrant, signal, assets = "Inflationary Boom",  "HOLD cautiously", ["gold", "energy", "commodities", "BTC"]
    elif not boom and inflationary: quadrant, signal, assets = "Stagflation",        "SELL rallies",    ["gold", "real_assets"]
    else:                           quadrant, signal, assets = "Deflationary Bust",  "REDUCE risk",     ["long_treasuries", "cash", "defensive"]

    # Gave's target allocations per quadrant (equities capped 80%, gold capped 30%)
    allocation_targets = {
        "Deflationary Boom":  {"equities": 70, "bonds": 20, "cash":  5, "gold":  5},
        "Inflationary Boom":  {"equities": 20, "bonds":  5, "cash": 20, "gold": 30},
        "Stagflation":        {"equities":  0, "bonds":  0, "cash": 70, "gold": 10},
        "Deflationary Bust":  {"equities": 10, "bonds": 70, "cash": 10, "gold": 10},
    }

    return {
        "quadrant":            quadrant,
        "tactical_signal":     signal,
        "recommended_assets":  assets,
        "allocation_target":   allocation_targets[quadrant],
        "growth": {
            "ratio":  gr["value"],
            "ma":     gr["ma"],
            "signal": "boom" if boom else "bust",
        },
        "inflation": {
            "ratio":  ir["value"],
            "ma":     ir["ma"],
            "signal": "inflationary" if inflationary else "deflationary",
        },
    }


# ── Wicksellian spread ────────────────────────────────────────────────────────
def get_wicksellian_signal(data: pd.DataFrame) -> dict:
    """
    Wicksellian spread proxy.

    Exact formula (Gave): (BAA_real_yield) - nominal_GDP_growth
      where BAA_real_yield = BAA_yield - 10yr_avg_CPI

    Thresholds:
      > 250 bps → recession imminent (predicted every recession since 1955)
      0-250 bps → normal credit cycle
      < 0       → stimulative / reflationary

    BAA yield and GDP are not available on yfinance.
    Proxy used here:
      market_rate  = 10Y yield normalised to its 3Y mean (↑ = restrictive)
      growth_proxy = Copper/Gold z-score (↑ = growth accelerating)
      spread_proxy = growth_proxy - market_rate  (unit: z-score difference)

    Positive proxy → stimulative; negative proxy → restrictive.
    Approximate 250-bps threshold is flagged when proxy < -1 std.
    """
    tnx    = _series(data, "^TNX")
    copper = _series(data, "HG=F")
    gold   = _series(data, "GC=F")

    df = pd.DataFrame({"tnx": tnx, "copper": copper, "gold": gold}).dropna()

    # Normalised 10Y yield vs its 3-year mean
    tnx_ratio = df["tnx"] / df["tnx"].rolling(3 * 252, min_periods=252).mean()

    # Copper/Gold z-score (252-day)
    cg    = df["copper"] / df["gold"]
    cg_z  = (cg - cg.rolling(252, min_periods=120).mean()) / \
             cg.rolling(252, min_periods=120).std().replace(0, np.nan)

    spread = cg_z - tnx_ratio

    val  = float(spread.iloc[-1])
    # flag if spread in deeply negative territory (proxy for >250bps in real spread)
    recession_flag = val < float(spread.rolling(252, min_periods=120).mean().iloc[-1]
                                 - spread.rolling(252, min_periods=120).std().iloc[-1])

    regime = "stimulative" if val > 0 else ("recession_warning" if recession_flag else "restrictive")

    return {
        "wicksellian_spread_proxy": round(val, 4),
        "regime":                   regime,
        "recession_warning":        recession_flag,
        "signal":                   "inflation/bubble risk" if val > 0 else
                                    "RECESSION IMMINENT (>250bps proxy)" if recession_flag else
                                    "deflationary pressure",
        "tnx_vs_3y_mean":           round(float(tnx_ratio.iloc[-1]), 4),
        "copper_gold_zscore":       round(float(cg_z.iloc[-1]), 4),
        "note":                     "proxy only; exact formula requires BAA yield + GDP growth (FRED)",
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
