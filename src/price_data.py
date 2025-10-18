import yfinance as yf
import pandas as pd
import numpy as np
import pytz, json, os, hashlib, pickle, time
from portfolio import PORTFOLIO_DICT, START_DATE, ASSET_TYPES

# Load tickers from Yahoo Finance API
def __load_tickers(tickers:list, interval:str = '1h', period:str = '1y', cache_duration:int = 3600):
    """
    Charge les données Yahoo Finance pour les tickers, avec cache local (pickle).
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5((str(tickers) + interval + period).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    cache_time_path = os.path.join(cache_dir, f"{cache_key}.time")

    # Vérifie si le cache existe et est encore valide
    if os.path.exists(cache_path) and os.path.exists(cache_time_path):
        with open(cache_time_path, "r") as f:
            cache_time = float(f.read())
        if time.time() - cache_time < cache_duration:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

 # Essaye de télécharger avec retries
    backoff = 1.0
    last_exception = None
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # logging.info(f"Downloading tickers (attempt {attempt})...")
            data = yf.download(tickers, interval=interval, period=period, auto_adjust=True, progress=False, threads=False)
            # Vérifie que la colonne 'Close' existe et contient quelque chose utile
            if data is None or data.empty:
                raise ValueError("yfinance returned empty data")
            # If multiindex, ensure at least one 'Close' column exists
            if ('Close' not in data.columns) and not any(isinstance(c, tuple) and c[0] == 'Close' for c in data.columns):
                raise ValueError("Downloaded data does not contain 'Close' column")
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            with open(cache_time_path, "w") as f:
                f.write(str(time.time()))
            return data
        except Exception as e:
            last_exception = e
            # logging.warning(f"Download attempt {attempt} failed: {e}")
            time.sleep(backoff)
            backoff *= 2


    return data

def _get_last_price(data, ticker, precision:int = 1):
    i = -1
    close_series = data['Close', ticker]
    while abs(i) <= len(close_series) and np.isnan(close_series.iloc[i]):
        i -= 1
    if abs(i) > len(close_series):
        raise ValueError(f"No valid price found for ticker {ticker}")
    price = close_series.iloc[i].round(precision)
    date = data.index[i]
    utc_timezone = pytz.timezone('UTC')
    paris_timezone = pytz.timezone('Europe/Paris')
    date_paris_timezone = date.replace(tzinfo=utc_timezone).astimezone(paris_timezone).strftime('%Y-%m-%d : %Hh%Mm%Ss')
    return date_paris_timezone, price


def _get_price_at_given_time(data, ticker, time, precision:int = 1):
    """
    Get the price of the ticker at a given time. Time value supported : '1d', '1mo', '1y', '5d' , '7d' , '3mo'
    """
    utc_timezone = pytz.timezone('UTC')

    dict = {'1d' : 1, '1mo' : 30, '1y' : 365, '5d' : 5, '7d' : 7, '3mo' : 90}
    i = -1
    while np.isnan(data['Close', ticker].iloc[i]): # Take the last true data (NaN during weekends)
        i += -1
    lastQuote = data.index[i]
    outputQuote = lastQuote - pd.Timedelta(days = dict[time])
    
    #Check if the given time is a valid one
    if outputQuote < data.index[0]:
        print("Invalid time")
        exit(1)
    
    i = 1
    while (data.index[-i] > outputQuote) or np.isnan(data['Close', ticker].iloc[-i]):
        i += 1
    price = data[('Close', ticker)].iloc[-i].round(precision)
    date = data.index[-i]
    utc_timezone = pytz.timezone('UTC')

    # Convert the UTC datetime to Paris time 
    paris_timezone = pytz.timezone('Europe/Paris')
    date_paris_timezone = date.replace(tzinfo=utc_timezone).astimezone(paris_timezone).strftime('%Y-%m-%d : %Hh%Mm%Ss')

    return date_paris_timezone, price

def _get_price_evolution(data, ticker, time, precision:int = 1):
    """
    Get the evolution of the price of the ticker at a given time. Time value supported : '1d', '1mo', '1y', '5d', '3mo'
    """
    _, lastPrice = _get_last_price(data, ticker, precision)
    _, previousPrice = _get_price_at_given_time(data, ticker, time, precision)
    rate = round((lastPrice - previousPrice) / previousPrice * 100, precision)
    return rate

def get_asset_section(data, ticker, precision=2, conversion_rate=1.0):
    _, last_price = _get_last_price(data, ticker, precision)
    change_1d = _get_price_evolution(data, ticker, '1d', precision)
    change_1mo = _get_price_evolution(data, ticker, '1mo', precision)
    return {
        "last_price": round(last_price * conversion_rate, precision),
        "change_1d_percent": change_1d,
        "change_1mo_percent": change_1mo,
    }

def get_portfolio_value_eur(data, portfolio:dict):
    """
    Calcule la valeur totale du portefeuille en EUR et la ventilation par actif.
    portfolio: dict avec {ticker: quantité}
    Retourne un dict avec la valeur totale et la ventilation.
    """
    _, eurusd_price = _get_last_price(data, 'EURUSD=X', precision=4)
    asset_values = {}
    total_value_eur = 0

    for ticker, quantity in portfolio.items():
        # Conversion USD->EUR si nécessaire
        if ticker == 'DBX9.DE':  # ChinaA en EUR, convertir en USD puis EUR
            _, last_price_eur = _get_last_price(data, ticker, precision=2)
            last_price_usd = last_price_eur * eurusd_price
            value_eur = last_price_usd * quantity / eurusd_price
        elif ticker in ['BTC-USD', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'CEMA.L', 'TTE']:  # USD assets
            _, last_price_usd = _get_last_price(data, ticker, precision=2)
            value_eur = last_price_usd * quantity / eurusd_price

        asset_values[ticker] = round(value_eur, 2)
        total_value_eur += value_eur

    return {
        "total_value_eur": round(total_value_eur, 2),
        "breakdown": asset_values
    }

def get_portfolio_performance_drilldown(data, portfolio:dict, start_date:str):
    """
    Retourne la performance depuis start_date et le rendement annualisé pour chaque type d'actif.
    Résultat formaté en JSON pour utilisation backend.
    """
    _, eurusd_price_now = _get_last_price(data, 'EURUSD=X', precision=4)
    drilldown = {}

    start_timestamp = pd.Timestamp(start_date)
    # helper: trouve le prix 'Close' le plus proche non-NaN autour d'un index
    def _find_nearest_valid_close(ticker, idx):
        try:
            series = data['Close', ticker]
        except Exception:
            return None
        n = len(series)
        if idx < 0 or idx >= n:
            return None
        # direct
        v = series.iloc[idx]
        if not np.isnan(v):
            return v
        # recherche symétrique
        for offset in range(1, max(idx+1, n-idx)):
            for cand in (idx - offset, idx + offset):
                if 0 <= cand < n:
                    val = series.iloc[cand]
                    if not np.isnan(val):
                        return val
        return None    
    # Pour chaque type d'actif, on cumule les valeurs
    type_values_now = {}
    type_values_start = {}
    total_now = 0.0
    total_start = 0.0
    skipped_tickers = []

    # nearest index for start timestamp
    idx_start = data.index.get_indexer([start_timestamp], method='nearest')[0]
    for ticker, quantity in portfolio.items():
        asset_type = ASSET_TYPES.get(ticker, 'other')
        # valeur actuelle (utilise _get_last_price pour robustesse)
        _, price_now = _get_last_price(data, ticker, precision=(4 if ticker == 'EURUSD=X' else 2))
        if np.isnan(price_now):
            # pas de prix actuel -> ignorer cet actif
            skipped_tickers.append(ticker)
            continue

        # conversion to EUR for current value
        elif ticker in ['BTC-USD', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'CEMA.L', 'TTE']:
            value_now_eur = price_now * quantity / eurusd_price_now
        else:  # tickers cotés en EUR
            value_now_eur = price_now * quantity

        total_now += value_now_eur
        type_values_now[asset_type] = type_values_now.get(asset_type, 0) + value_now_eur

        # valeur au start_date : chercher prix non-NaN proche de idx_start
        price_start = _find_nearest_valid_close(ticker, idx_start)
        # taux EURUSD au start (utilisé pour convertir USD->EUR pour la valeur de départ)
        eurusd_price_start = _find_nearest_valid_close('EURUSD=X', idx_start)

        if price_start is None or np.isnan(price_start):
            skipped_tickers.append(ticker)
            continue

        # conversion to EUR for start value
        elif ticker in ['BTC-USD', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'CEMA.L', 'TTE']:
            if eurusd_price_start is None or np.isnan(eurusd_price_start) or eurusd_price_start == 0:
                skipped_tickers.append(ticker)
                continue
            value_start_eur = price_start * quantity / eurusd_price_start
        else:
            value_start_eur = price_start * quantity

        total_start += value_start_eur
        type_values_start[asset_type] = type_values_start.get(asset_type, 0) + value_start_eur

    # Calcul des performances par type d'actif (existante)
    for asset_type, value_now in type_values_now.items():
        value_start = type_values_start.get(asset_type, 0)
        if value_start > 0:
            perf_total = round((value_now - value_start) / value_start * 100, 2)
            days = max(1, (data.index[-1] - start_timestamp).days)
            annualized_return = round(((value_now / value_start) ** (365 / days) - 1) * 100, 2)
        else:
            perf_total = None
            annualized_return = None

        drilldown[asset_type] = {
            "performance_since_start_percent": perf_total,
            "annualized_return_percent": annualized_return
        }

    # Calcul performance totale du portefeuille
    if total_start > 0:
        perf_total_portfolio = round((total_now - total_start) / total_start * 100, 2)
        days_total = max(1, (data.index[-1] - start_timestamp).days)
        annualized_return_portfolio = round(((total_now / total_start) ** (365 / days_total) - 1) * 100, 2)
    else:
        perf_total_portfolio = None
        annualized_return_portfolio = None

    return {
        "total": {
            "total_value_now_eur": round(total_now, 2),
            "total_value_start_eur": round(total_start, 2),
            "performance_since_start_percent": perf_total_portfolio,
            "annualized_return_percent": annualized_return_portfolio,
            "skipped_tickers": sorted(set(skipped_tickers))
        },
        "by_type": drilldown
    }

def get_portfolio_allocation_by_type(data, portfolio:dict):
    """
    Retourne la répartition du portefeuille par type de classe d'actifs (en EUR).
    """
    _, current_eurusd_price = _get_last_price(data, 'EURUSD=X', precision=4)
    allocation = {}
    total_value_eur = 0

    for ticker, quantity in portfolio.items():
        asset_type = ASSET_TYPES.get(ticker, 'other')
        # Calcul de la valeur en EUR
        if ticker == 'DBX9.DE':
            _, last_price_eur = _get_last_price(data, ticker, precision=2)
            value_eur = last_price_eur * quantity
        elif ticker in ['BTC-USD', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'CEMA.L', 'TTE']:
            _, last_price_usd = _get_last_price(data, ticker, precision=2)
            value_eur = last_price_usd * quantity / current_eurusd_price

        allocation[asset_type] = allocation.get(asset_type, 0) + value_eur
        total_value_eur += value_eur

    # Formatage pour affichage en pourcentage
    allocation_percent = {k: round(v / total_value_eur * 100, 2) for k, v in allocation.items()}

    return {
        "allocation_eur": {k: round(v, 2) for k, v in allocation.items()},
        "allocation_percent": allocation_percent,
        "total_value_eur": round(total_value_eur, 2)
    }

if __name__ == "__main__":
    data = __load_tickers(['BTC-USD','EURUSD=X', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'DBX9.DE', 'CEMA.L', 'TTE'], interval = '1d', period = '2y')
    _, current_eurusd_price = _get_last_price(data, 'EURUSD=X', precision = 4)
    output = {
        "BTC_USD": get_asset_section(data, 'BTC-USD', precision=2),
        "EUR_USD": get_asset_section(data, 'EURUSD=X', precision=4),
        "GOLD_USD": get_asset_section(data, 'GC=F', precision=2),
        "ENERGY_USD": get_asset_section(data, 'XDW0L.XC', precision=2),
        "TOTAL_ENERGY_USD": get_asset_section(data, 'TTE', precision=2),
        "HKTech_USD": get_asset_section(data, 'HSTE.L', precision=2),
        "ChinaA_USD": get_asset_section(data, 'DBX9.DE', precision=2, conversion_rate=current_eurusd_price),
        "EmergingMarkets_USD": get_asset_section(data, 'CEMA.L', precision=2)
    }
    # portfolio_value = get_portfolio_value_eur(data, PORTFOLIO_DICT)
    # print(json.dumps(portfolio_value, indent=2))
    result = {
    "assets": output,
    "performance_by_type": get_portfolio_performance_drilldown(data, PORTFOLIO_DICT, start_date=START_DATE),
    "allocation_percent": get_portfolio_allocation_by_type(data, PORTFOLIO_DICT)["allocation_percent"]
    }
    print(json.dumps(result, indent=2))

