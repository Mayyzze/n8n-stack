import json
from portfolio import PORTFOLIO_DICT, START_DATE, EVENTS
from fetcher import load_tickers
from prices import get_last_price
from analytics import get_asset_section, get_portfolio_performance_drilldown, get_portfolio_allocation_by_type, get_mwr_by_type

if __name__ == "__main__":
    data = load_tickers(
        ['BTC-USD', 'EURUSD=X', 'GC=F', 'CL=F', 'XDW0L.XC', 'HSTE.L', 'DBX9.DE', 'CEMA.L', 'TTE', 'MSTR'],
        interval='1d',
        period='2y'
    )
    if data is None: 
        raise ValueError('Unexpected error in load_tickers')
    _, current_eurusd_price = get_last_price(data, 'EURUSD=X', precision=4)
    output = {
        "GOLD_USD": get_asset_section(data, 'GC=F', precision=2),
        "WTI_USD": get_asset_section(data, 'CL=F', precision=2),
        "EUR_USD": get_asset_section(data, 'EURUSD=X', precision=4),
        "BTC_USD": get_asset_section(data, 'BTC-USD', precision=2),
        "ENERGY_USD": get_asset_section(data, 'XDW0L.XC', precision=2),
        "TOTAL_ENERGY_USD": get_asset_section(data, 'TTE', precision=2),
        "HKTech_USD": get_asset_section(data, 'HSTE.L', precision=2),
        "ChinaA_USD": get_asset_section(data, 'DBX9.DE', precision=2, conversion_rate=current_eurusd_price),
        "EmergingMarkets_USD": get_asset_section(data, 'CEMA.L', precision=2),
        "MSTR_USD": get_asset_section(data, 'MSTR', precision=2)
    }
    drilldown = get_portfolio_performance_drilldown(data, PORTFOLIO_DICT, start_date=START_DATE)
    mwr       = get_mwr_by_type(data, EVENTS)
    for atype, mwr_val in mwr.items():
        drilldown.setdefault(atype, {})['mwr_annualized_percent'] = mwr_val

    result = {
        "assets": output,
        "performance_by_type": drilldown,
        "allocation_percent": get_portfolio_allocation_by_type(data, PORTFOLIO_DICT)["allocation_percent"],
    }
    print(json.dumps(result, indent=2))
