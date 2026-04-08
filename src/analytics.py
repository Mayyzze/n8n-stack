import numpy as np
import pandas as pd
from scipy.optimize import brentq # type: ignore
from portfolio import ASSET_TYPES, TICKER_CURRENCY
from prices import get_last_price, get_valid_price_at_idx, get_price_evolution


def _to_eur(amount: float, ticker: str, eurusd: float) -> float:
    """Converts a native-currency amount to EUR using the provided EURUSD rate."""
    return amount if TICKER_CURRENCY.get(ticker) == 'EUR' else amount / eurusd


def _perf_entry(value_now: float, value_start: float, days: int) -> dict:
    """Returns cumulative and annualized performance metrics between two values."""
    if value_start <= 0:
        return {"performance_since_start_percent": None, "twr_annualized_percent": None}
    cumulative  = round((value_now - value_start) / value_start * 100, 2)
    annualized  = round(((value_now / value_start) ** (365 / max(1, days)) - 1) * 100, 2)
    return {"performance_since_start_percent": cumulative, "twr_annualized_percent": annualized}


def _ticker_value_eur(data: pd.DataFrame, ticker: str, quantity: float, eurusd: float) -> float:
    """Returns the current EUR value of a position."""
    _, price = get_last_price(data, ticker, precision=2)
    return _to_eur(price * quantity, ticker, eurusd)


def get_asset_section(data: pd.DataFrame, ticker: str, precision: int = 2, conversion_rate: float = 1.0) -> dict:
    _, last_price = get_last_price(data, ticker, precision)
    return {
        "last_price":        round(last_price * conversion_rate, precision),
        "change_1d_percent":  get_price_evolution(data, ticker, '1d',  precision),
        "change_1mo_percent": get_price_evolution(data, ticker, '1mo', precision),
    }


def get_portfolio_value_eur(data: pd.DataFrame, portfolio: dict) -> dict:
    """Returns total EUR value and per-ticker breakdown of a portfolio."""
    _, eurusd = get_last_price(data, 'EURUSD=X', precision=4)
    breakdown: dict[str, float] = {}
    for ticker, qty in portfolio.items():
        breakdown[ticker] = round(_ticker_value_eur(data, ticker, qty, eurusd), 2)
    total = sum(breakdown.values())
    return {"total_value_eur": round(total, 2), "breakdown": breakdown}


def get_portfolio_performance_drilldown(data: pd.DataFrame, portfolio: dict, start_date: str) -> dict:
    """
    Returns cumulative and TWR-annualized performance since start_date,
    broken down by asset type and globally.
    """
    _, eurusd_now   = get_last_price(data, 'EURUSD=X', precision=4)
    start_ts        = pd.Timestamp(start_date)
    idx_start       = data.index.get_indexer(pd.Index([start_ts]), method='nearest')[0]
    eurusd_start    = get_valid_price_at_idx(data, 'EURUSD=X', idx_start) or eurusd_now
    days            = (data.index[-1] - start_ts).days

    type_now:   dict[str, float] = {}
    type_start: dict[str, float] = {}

    for ticker, qty in portfolio.items():
        atype       = ASSET_TYPES.get(ticker, 'other')
        _, price_now = get_last_price(data, ticker, precision=2)
        if np.isnan(price_now):
            continue

        type_now[atype] = type_now.get(atype, 0.0) + _to_eur(price_now * qty, ticker, eurusd_now)

        price_start = get_valid_price_at_idx(data, ticker, idx_start)
        if price_start is None or np.isnan(price_start):
            continue
        if TICKER_CURRENCY.get(ticker) != 'EUR' and (not eurusd_start or eurusd_start == 0):
            continue

        type_start[atype] = type_start.get(atype, 0.0) + _to_eur(price_start * qty, ticker, eurusd_start)

    output = {
        atype: _perf_entry(type_now[atype], type_start.get(atype, 0.0), days)
        for atype in type_now
    }
    output['Global'] = _perf_entry(sum(type_now.values()), sum(type_start.values()), days)
    return output


def get_portfolio_allocation_by_type(data: pd.DataFrame, portfolio: dict) -> dict:
    """Returns portfolio allocation in EUR and percent, grouped by asset type."""
    value_data = get_portfolio_value_eur(data, portfolio)
    total      = value_data['total_value_eur']
    allocation: dict[str, float] = {}
    for ticker, value_eur in value_data['breakdown'].items():
        atype = ASSET_TYPES.get(ticker, 'other')
        allocation[atype] = allocation.get(atype, 0.0) + value_eur
    return {
        "allocation_eur":     {k: round(v, 2) for k, v in allocation.items()},
        "allocation_percent": {k: round(v / total * 100, 2) for k, v in allocation.items()},
        "total_value_eur":    total,
    }


def _solve_irr(cash_flows: list[tuple[int, float]]) -> float | None:
    """
    Finds IRR by solving NPV(r) = 0 numerically via brentq.
    Upper bound is 5000% to handle crypto-level returns.
    Returns None when brentq cannot bracket a root (e.g. all buy prices are 0).
    """
    def npv(r):
        return sum(cf / (1 + r) ** (t / 365) for t, cf in cash_flows)
    try:
        return round(brentq(npv, -0.99, 50.0) * 100, 2)
    except ValueError:
        return None


def get_mwr_by_type(data: pd.DataFrame, events: list[dict]) -> dict[str, float | None]:
    """
    Computes Money-Weighted Return (IRR) per asset type and globally.

    Cash flows per type in EUR:
      - buys  → negative (money out)
      - sells → positive (money in)
      - terminal value → current mark-to-market (appended at the end)
    """
    start_date = pd.Timestamp(events[0]['date'])
    t_terminal = (data.index[-1] - start_date).days

    type_cash_flows: dict[str, list[tuple[int, float]]] = {}
    type_holdings:   dict[str, dict[str, float]]        = {}
    global_cash_flows: list[tuple[int, float]]          = []

    for event in events:
        t      = (pd.Timestamp(event['date']) - start_date).days
        eurusd = event['eurusd']
        type_net: dict[str, float] = {}
        global_net = 0.0

        for ticker, lot in event.get('buys', {}).items():
            amount_eur = _to_eur(lot['qty'] * lot['price'], ticker, eurusd)
            atype      = ASSET_TYPES.get(ticker, 'other')
            type_net[atype] = type_net.get(atype, 0.0) - amount_eur
            global_net -= amount_eur
            type_holdings.setdefault(atype, {})
            type_holdings[atype][ticker] = type_holdings[atype].get(ticker, 0.0) + lot['qty']

        for ticker, lot in event.get('sells', {}).items():
            amount_eur = _to_eur(lot['qty'] * lot['price'], ticker, eurusd)
            atype      = ASSET_TYPES.get(ticker, 'other')
            type_net[atype] = type_net.get(atype, 0.0) + amount_eur
            global_net += amount_eur
            type_holdings.setdefault(atype, {})
            type_holdings[atype][ticker] = type_holdings[atype].get(ticker, 0.0) - lot['qty']

        for atype, net in type_net.items():
            type_cash_flows.setdefault(atype, []).append((t, net))
        global_cash_flows.append((t, global_net))

    result: dict[str, float | None] = {}
    global_terminal = 0.0

    for atype, holdings in type_holdings.items():
        terminal = get_portfolio_value_eur(data, holdings)['total_value_eur']
        global_terminal += terminal
        cfs = type_cash_flows.get(atype, []) + [(t_terminal, terminal)]
        result[atype] = _solve_irr(cfs)

    result['Global'] = _solve_irr(global_cash_flows + [(t_terminal, global_terminal)])
    return result