"""
Gavekal / Charles Gave macro indicators.

Framework: Four Quadrants (growth vs inflation axes) → regime detection → asset allocation.
Ref: "The General Theory of Portfolio Construction" - Charles Gave, Gavekal Research.
     Institut des Libertés (institutdeslibertes.org) — sound money / libertarian economics.

Economic Quadrants (2×2):
  X-axis (growth):    S&P 500 / WTI  vs 7-year SMA
  Y-axis (inflation): Gold / 10Y Treasury ETF vs 7-year SMA

  Disinflationary Boom  (right + bottom): best regime; buy dips on equities
  Inflationary Boom     (right + top)   : real assets, energy, commodities
  Deflationary Bust     (left  + bottom): long Treasuries; reduce risk
  Stagflation           (left  + top)   : worst regime; sell rallies; cash/gold

Monetary Quadrants (2×2, separate financial-cycle layer):
  X-axis: Yield curve slope  (10Y − 3M: positive = normal, negative = inverted)
  Y-axis: Short-rate momentum (13W T-bill vs 90-day MA: rising or falling)

  Goldilocks    (+ slope + falling rates): equities and bonds both favoured
  Reflation     (+ slope + rising rates) : equities OK; bonds under pressure
  QE / Easing   (− slope + falling rates): long bonds rally; cash > equities
  Credit Crunch (− slope + rising rates) : worst monetary environment; flee to gold/cash

Wicksellian spread (exact): BAA_real_yield − nominal_GDP_growth
  < 0        → stimulative / reflationary
  0–250 bps  → normal credit cycle (capitalism works; banks fairly rewarded)
  > 250 bps  → recession imminent (predicted every recession since 1955)
  > 600 bps  → deep recession / financial crisis

Gave's MV=PQ identity: V (velocity) is an independent variable, not a residual.
GVI (Gavekal Velocity Indicator): market proxy that leads actual velocity by ~6 months.

Institut des Libertés pillars:
  Sound money, rule of law, property rights, critique of central bank overreach.
  Gold (and to some degree BTC) as protection against fiat debasement.

Note: BAA yield and GDP growth are not available via yfinance.
Market-based proxies are used throughout (documented per function).
"""

import json, os, hashlib, pickle, time
import numpy as np
import pandas as pd
import yfinance as yf

# ── tickers ──────────────────────────────────────────────────────────────────
INDICATOR_TICKERS = [
    "^GSPC",      # S&P 500 (growth proxy)
    "CL=F",       # WTI crude (growth / commodity)
    "GC=F",       # Gold (inflation / safe-haven / sound money)
    "HG=F",       # Copper (global growth barometer)
    "TLT",        # 20Y Treasury ETF (deflation hedge, reference)
    "^TNX",       # 10Y yield
    "^IRX",       # 13W yield (short-rate proxy)
    "DX-Y.NYB",   # USD index (fiat quality)
    "IEF",        # 7-10Y Treasury ETF (Gold/Treasury inflation ratio)
    "BTC-USD",    # Bitcoin (hard-money competition vs gold)
    "EURUSD=X",   # EUR/USD (for gold-in-EUR sound money score)
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


# ── Monetary Quadrants ────────────────────────────────────────────────────────
def get_monetary_quadrants(data: pd.DataFrame) -> dict:
    """
    Gave's financial-cycle layer: yield curve slope × short-rate momentum.

    Axis 1 — Yield curve slope: 10Y − 3M (^TNX − ^IRX)
      positive → normal (banks willing to lend)
      negative → inverted (credit tightening signal)

    Axis 2 — Short-rate momentum: 13W T-bill vs its 90-day MA
      falling → monetary easing / accommodation
      rising  → monetary tightening

    4 monetary regimes and their asset implications:
      Goldilocks    (+ slope, falling rates): both equities and bonds do well
      Reflation     (+ slope, rising rates) : equities OK; bonds under pressure; real assets
      QE / Easing   (− slope, falling rates): long bonds outperform; cash > equities
      Credit Crunch (− slope, rising rates) : worst environment; flee to gold and cash
    """
    tnx = _series(data, "^TNX")
    irx = _series(data, "^IRX")

    df = pd.DataFrame({"tnx": tnx, "irx": irx}).dropna()
    df["slope"] = df["tnx"] - df["irx"]
    df["irx_ma90"] = df["irx"].rolling(90, min_periods=45).mean()

    slope_positive  = float(df["slope"].iloc[-1]) > 0
    short_rate_rising = float(df["irx"].iloc[-1]) > float(df["irx_ma90"].iloc[-1])

    slope_val   = round(float(df["slope"].iloc[-1]) * 100, 1)   # in bps
    irx_val     = round(float(df["irx"].iloc[-1]), 3)
    irx_ma_val  = round(float(df["irx_ma90"].iloc[-1]), 3)

    if   slope_positive and not short_rate_rising:
        regime, assets, signal = "Goldilocks",    ["equities", "bonds", "growth"],       "HOLD / BUY equities and bonds"
    elif slope_positive and     short_rate_rising:
        regime, assets, signal = "Reflation",     ["equities", "real_assets", "energy"], "OVERWEIGHT real assets; trim bonds"
    elif not slope_positive and not short_rate_rising:
        regime, assets, signal = "QE / Easing",   ["long_bonds", "cash"],                "BUY long bonds; reduce equity beta"
    else:
        regime, assets, signal = "Credit Crunch", ["gold", "cash", "defensive"],         "REDUCE risk; accumulate gold and cash"

    return {
        "monetary_regime":      regime,
        "tactical_signal":      signal,
        "recommended_assets":   assets,
        "yield_curve": {
            "slope_10y_3m_bp": slope_val,
            "inverted":        not slope_positive,
        },
        "short_rate": {
            "irx":       irx_val,
            "ma_90d":    irx_ma_val,
            "trend":     "rising" if short_rate_rising else "falling",
        },
    }


# ── Velocity of Money Proxy (GVI-like) ────────────────────────────────────────
def get_velocity_indicator(data: pd.DataFrame) -> dict:
    """
    Gavekal Velocity Indicator (GVI) proxy.

    Gave's key insight: in MV=PQ, velocity V is an *independent* variable (not
    the tautological residual). The GVI uses market data to lead actual velocity
    by ~6 months — signalling whether cash held by economic agents is starting
    to move into the real economy.

    Proxy construction (market-based, all from yfinance):
      velocity_proxy = copper/gold 6-month momentum + yield-curve steepness

      • Copper/Gold 6m momentum (CuAu_mom): industrial demand vs safe-haven
        demand, normalised to 252-day z-score. Rising → money circulating faster.
      • Yield curve component (YC): (10Y − 3M) normalised to 252-day range [0,1].
        Steeper → banks more willing to lend → higher velocity ahead.

      GVI_proxy = 0.6 × CuAu_mom + 0.4 × YC_norm

    Thresholds (z-score based):
      > +0.5  → velocity accelerating → inflationary pressure in 6 months
      −0.5 to +0.5 → stable velocity → neutral
      < −0.5  → velocity collapsing → deflationary threat ahead
    """
    tnx    = _series(data, "^TNX")
    irx    = _series(data, "^IRX")
    copper = _series(data, "HG=F")
    gold   = _series(data, "GC=F")

    df = pd.DataFrame({"tnx": tnx, "irx": irx, "copper": copper, "gold": gold}).dropna()

    # Copper/Gold 6-month momentum, z-scored over 252 days
    cg     = df["copper"] / df["gold"]
    cg_6m  = cg.pct_change(126)                                         # 6-month return
    cg_z   = (cg_6m - cg_6m.rolling(252, min_periods=120).mean()) / \
              cg_6m.rolling(252, min_periods=120).std().replace(0, np.nan)

    # Yield curve normalised to rolling [0,1] over 252 days
    slope  = df["tnx"] - df["irx"]
    s_min  = slope.rolling(252, min_periods=120).min()
    s_max  = slope.rolling(252, min_periods=120).max()
    yc_norm = (slope - s_min) / (s_max - s_min).replace(0, np.nan)

    gvi    = 0.6 * cg_z + 0.4 * yc_norm
    val    = float(gvi.iloc[-1])

    if   val > 0.5:  regime, signal = "accelerating", "velocity rising → inflation/reflation in ~6 months"
    elif val < -0.5: regime, signal = "collapsing",   "velocity falling → deflationary threat in ~6 months"
    else:            regime, signal = "stable",        "velocity stable → no major regime shift imminent"

    return {
        "gvi_proxy":              round(val, 4),
        "velocity_regime":        regime,
        "6m_forward_signal":      signal,
        "copper_gold_6m_zscore":  round(float(cg_z.iloc[-1]), 4),
        "yield_curve_norm":       round(float(yc_norm.iloc[-1]), 4),
        "note": "proxy only; true GVI requires proprietary Gavekal data",
    }


# ── Sound Money Score ─────────────────────────────────────────────────────────
def get_sound_money_score(data: pd.DataFrame) -> dict:
    """
    Institut des Libertés 'sound money' debasement score.

    Charles Gave argues that central bank overreach (prolonged low rates, QE)
    debases fiat money. Gold rising in *all* major currencies is the clearest
    signal of system-wide debasement.

    Score components (each +1 if triggered):
      1. Gold above its 52-week MA in USD       → USD fiat losing value
      2. Gold above its 52-week MA in EUR       → EUR fiat losing value
         (Gold_EUR = Gold_USD / EURUSD)
      3. DXY below its 52-week MA               → USD structurally weakening

    Score 0: hard assets unloved; fiat trusted; sound monetary environment
    Score 1: mild debasement signal
    Score 2: moderate debasement; diversify into real assets
    Score 3: maximum alarm — gold rising in all currencies + weak USD
              → Gave's prescription: maximum gold allocation
    """
    gold   = _series(data, "GC=F")
    eurusd = _series(data, "EURUSD=X")
    dxy    = _series(data, "DX-Y.NYB")

    df = pd.DataFrame({"gold": gold, "eurusd": eurusd, "dxy": dxy}).dropna()

    # Gold in EUR
    df["gold_eur"] = df["gold"] / df["eurusd"]

    ma52 = 252   # ~52 weeks of trading days

    gold_usd_ma  = df["gold"].rolling(ma52, min_periods=120).mean()
    gold_eur_ma  = df["gold_eur"].rolling(ma52, min_periods=120).mean()
    dxy_ma       = df["dxy"].rolling(ma52, min_periods=120).mean()

    gold_usd_above = float(df["gold"].iloc[-1])    > float(gold_usd_ma.iloc[-1])
    gold_eur_above = float(df["gold_eur"].iloc[-1]) > float(gold_eur_ma.iloc[-1])
    dxy_below      = float(df["dxy"].iloc[-1])     < float(dxy_ma.iloc[-1])

    score = int(gold_usd_above) + int(gold_eur_above) + int(dxy_below)

    if   score == 3: interpretation = "maximum debasement alarm — accumulate gold, real assets"
    elif score == 2: interpretation = "significant debasement — increase hard asset exposure"
    elif score == 1: interpretation = "mild debasement signal — monitor"
    else:            interpretation = "sound monetary environment — fiat trusted"

    return {
        "sound_money_score":   score,           # 0 (sound) → 3 (maximum alarm)
        "interpretation":      interpretation,
        "components": {
            "gold_above_52w_ma_usd": gold_usd_above,
            "gold_above_52w_ma_eur": gold_eur_above,
            "dxy_below_52w_ma":      dxy_below,
        },
        "values": {
            "gold_usd":     round(float(df["gold"].iloc[-1]), 2),
            "gold_usd_ma":  round(float(gold_usd_ma.iloc[-1]), 2),
            "gold_eur":     round(float(df["gold_eur"].iloc[-1]), 2),
            "gold_eur_ma":  round(float(gold_eur_ma.iloc[-1]), 2),
            "dxy":          round(float(df["dxy"].iloc[-1]), 2),
            "dxy_ma":       round(float(dxy_ma.iloc[-1]), 2),
        },
    }


# ── BTC / Gold ratio ──────────────────────────────────────────────────────────
def get_btc_gold_ratio(data: pd.DataFrame) -> dict:
    """
    Hard-money competition: Bitcoin vs Gold.

    Gave has noted Bitcoin's role as a competing store of value (Institut des
    Libertés: 'Bitcoin et Big Brother'). Monitoring BTC/Gold tells us which
    form of 'hard money' the market is currently preferring.

    BTC/Gold rising  → market prefers BTC as store of value (risk-on hard money;
                        typically aligned with Inflationary Boom quadrant)
    BTC/Gold falling → market prefers gold (risk-off hard money;
                        typical during Deflationary Bust or Stagflation)

    Signals use 200-day MA (risk-asset standard) and 52-week MA.
    """
    btc  = _series(data, "BTC-USD")
    gold = _series(data, "GC=F")

    df = pd.DataFrame({"btc": btc, "gold": gold}).dropna()
    df["btc_gold"] = df["btc"] / df["gold"]

    ma200 = df["btc_gold"].rolling(200, min_periods=100).mean()
    ma52w = df["btc_gold"].rolling(252, min_periods=120).mean()

    val    = float(df["btc_gold"].iloc[-1])
    ma200v = float(ma200.iloc[-1])
    ma52wv = float(ma52w.iloc[-1])

    above_200 = val > ma200v
    above_52w = val > ma52wv

    # 90-day momentum
    mom90 = float(df["btc_gold"].iloc[-1] - df["btc_gold"].iloc[-min(90, len(df)-1)])
    mom90_pct = round(mom90 / ma200v * 100, 2) if ma200v else None

    if   above_200 and above_52w: preference, signal = "BTC", "risk-on hard money — BTC outperforming gold"
    elif not above_200 and not above_52w: preference, signal = "Gold", "risk-off hard money — gold preferred over BTC"
    else:                                 preference, signal = "Neutral", "transitional — no clear hard-money preference"

    return {
        "btc_gold_ratio":        round(val, 4),
        "ma_200d":               round(ma200v, 4),
        "ma_52w":                round(ma52wv, 4),
        "above_ma200":           above_200,
        "above_ma52w":           above_52w,
        "momentum_90d_pct":      mom90_pct,
        "hard_money_preference": preference,
        "signal":                signal,
    }


# ── Regime Transition Risk ────────────────────────────────────────────────────
def get_regime_transition_risk(data: pd.DataFrame) -> dict:
    """
    Early warning: how close are the Four Quadrant ratios to crossing their MAs?

    Gave's quadrant signals flip when the ratio crosses its 7-year MA. A ratio
    within ±5% of the MA is in the 'transition zone' — the next data points
    could shift the regime.

    Also detects recent transitions (ratio crossed its MA in the last 90 days).

    Outputs:
      growth_distance_pct   : % gap between S&P/WTI ratio and its 7Y MA
                              (+ve = boom, −ve = bust)
      inflation_distance_pct: % gap between Gold/IEF ratio and its 7Y MA
                              (+ve = inflationary, −ve = deflationary)
      transition_zone       : True if either distance is within ±5%
      recent_transition     : True if either ratio crossed its MA in last 90 days
      risk_level            : 'high' | 'elevated' | 'low'
    """
    spx  = _series(data, "^GSPC")
    wti  = _series(data, "CL=F")
    gold = _series(data, "GC=F")
    ief  = _series(data, "IEF")

    window = MA_YEARS * 252

    df = pd.DataFrame({"spx": spx, "wti": wti, "gold": gold, "ief": ief}).dropna()
    df["gr"]   = df["spx"] / df["wti"]
    df["ir"]   = df["gold"] / df["ief"]
    df["gr_ma"] = df["gr"].rolling(window, min_periods=window // 2).mean()
    df["ir_ma"] = df["ir"].rolling(window, min_periods=window // 2).mean()

    gr_now  = float(df["gr"].iloc[-1])
    ir_now  = float(df["ir"].iloc[-1])
    gr_ma   = float(df["gr_ma"].iloc[-1])
    ir_ma   = float(df["ir_ma"].iloc[-1])

    gr_dist_pct = round((gr_now - gr_ma) / gr_ma * 100, 2)
    ir_dist_pct = round((ir_now - ir_ma) / ir_ma * 100, 2)

    # Transition zone: within ±5%
    in_zone = abs(gr_dist_pct) < 5.0 or abs(ir_dist_pct) < 5.0

    # Recent transition: did the sign of (ratio − MA) change in the last 90 days?
    lookback = min(90, len(df) - 1)
    gr_sign_now  = np.sign(gr_now - gr_ma)
    ir_sign_now  = np.sign(ir_now - ir_ma)
    gr_sign_past = np.sign(float(df["gr"].iloc[-lookback]) - float(df["gr_ma"].iloc[-lookback]))
    ir_sign_past = np.sign(float(df["ir"].iloc[-lookback]) - float(df["ir_ma"].iloc[-lookback]))
    recent_transition = (gr_sign_now != gr_sign_past) or (ir_sign_now != ir_sign_past)

    if   in_zone and recent_transition: risk = "high"
    elif in_zone or recent_transition:  risk = "elevated"
    else:                               risk = "low"

    return {
        "risk_level":             risk,
        "transition_zone":        in_zone,
        "recent_transition_90d":  recent_transition,
        "growth_axis": {
            "ratio":        round(gr_now, 4),
            "ma_7y":        round(gr_ma, 4),
            "distance_pct": gr_dist_pct,
            "side":         "boom" if gr_dist_pct > 0 else "bust",
        },
        "inflation_axis": {
            "ratio":        round(ir_now, 4),
            "ma_7y":        round(ir_ma, 4),
            "distance_pct": ir_dist_pct,
            "side":         "inflationary" if ir_dist_pct > 0 else "deflationary",
        },
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
    """
    Fetch data once and return all Gavekal / Institut des Libertés indicators.

    Economic layer  : Four Quadrants + Wicksellian spread + Copper/Gold
    Monetary layer  : Monetary Quadrants + Yield Curve + Dollar Strength
    Velocity layer  : GVI proxy (leads ~6 months)
    Sound money     : Debasement score (gold in USD + EUR + DXY)
    Hard money race : BTC/Gold ratio
    Risk management : Regime Transition Risk (boundary proximity)
    Portfolio fit   : Per-ticker favourability vs current quadrant
    """
    data = _load_indicator_data(period="max", interval="1d")

    four_q    = get_four_quadrants(data)
    monetary  = get_monetary_quadrants(data)

    return {
        # ── Economic quadrants (Gave's primary framework) ──
        "four_quadrants":          four_q,
        "wicksellian_signal":      get_wicksellian_signal(data),
        "copper_gold_ratio":       get_copper_gold_ratio(data),

        # ── Monetary / financial cycle layer ──
        "monetary_quadrants":      monetary,
        "yield_curve":             get_yield_curve(data),
        "dollar_strength":         get_dollar_strength(data),

        # ── Velocity of money (GVI proxy) ──
        "velocity_indicator":      get_velocity_indicator(data),

        # ── Institut des Libertés: sound money ──
        "sound_money_score":       get_sound_money_score(data),

        # ── Hard money competition ──
        "btc_gold_ratio":          get_btc_gold_ratio(data),

        # ── Regime transition early warning ──
        "regime_transition_risk":  get_regime_transition_risk(data),

        # ── Portfolio positioning vs current regime ──
        "portfolio_fit": {
            "economic_quadrant": four_q["quadrant"],
            "monetary_regime":   monetary["monetary_regime"],
            "ticker_fit":        get_portfolio_quadrant_fit(four_q["quadrant"]),
        },
    }


if __name__ == "__main__":
    result = get_all_gavekal_indicators()
    print(json.dumps(result, indent=2))
