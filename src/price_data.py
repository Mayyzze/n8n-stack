import yfinance as yf
import pandas as pd
import numpy as np
import pytz, json
from portfolio import PORTFOLIO_DICT

# Load tickers from Yahoo Finance API
def load_tickers(tickers:list, interval:str = '1h', period:str = '1y'):
    data = yf.download(tickers, interval = interval, period =period, auto_adjust=True, progress=False, threads=True)
    #print(data.tail(20))
    return data

def get_last_price(data, ticker, precision:int = 1):
    i = -1
    while np.isnan(data['Close', ticker].iloc[i]): # Take the last true data (NaN during weekends)
        i += -1
    price = data[('Close', ticker)].iloc[i].round(precision)
    date = data.index[i]
    utc_timezone = pytz.timezone('UTC')

    # Convert the UTC datetime to Paris time 
    paris_timezone = pytz.timezone('Europe/Paris')
    date_paris_timezone = date.replace(tzinfo=utc_timezone).astimezone(paris_timezone).strftime('%Y-%m-%d : %Hh%Mm%Ss')
    return date_paris_timezone, price


def get_price_at_given_time(data, ticker, time, precision:int = 1):
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

def get_price_evolution(data, ticker, time, precision:int = 1):
    """
    Get the evolution of the price of the ticker at a given time. Time value supported : '1d', '1mo', '1y', '5d', '3mo'
    """
    _, lastPrice = get_last_price(data, ticker, precision)
    _, previousPrice = get_price_at_given_time(data, ticker, time, precision)
    rate = round((lastPrice - previousPrice) / previousPrice * 100, precision)
    return rate

def get_asset_section(data, ticker, precision=2, conversion_rate=1.0):
    _, last_price = get_last_price(data, ticker, precision)
    change_1d = get_price_evolution(data, ticker, '1d', precision)
    change_1mo = get_price_evolution(data, ticker, '1mo', precision)
    change_1y = get_price_evolution(data, ticker, '1y', precision)
    return {
        "last_price": round(last_price * conversion_rate, precision),
        "change_1d_percent": change_1d,
        "change_1mo_percent": change_1mo,
        "change_1y_percent": change_1y
    }

def get_portfolio_value_eur(data, portfolio:dict):
    """
    Calcule la valeur totale du portefeuille en EUR et la ventilation par actif.
    portfolio: dict avec {ticker: quantité}
    Retourne un dict avec la valeur totale et la ventilation.
    """
    _, eurusd_price = get_last_price(data, 'EURUSD=X', precision=4)
    asset_values = {}
    total_value_eur = 0

    for ticker, quantity in portfolio.items():
        # Conversion USD->EUR si nécessaire
        if ticker == 'DBX9.DE':  # ChinaA en EUR, convertir en USD puis EUR
            _, last_price_eur = get_last_price(data, ticker, precision=2)
            last_price_usd = last_price_eur * eurusd_price
            value_eur = last_price_usd * quantity / eurusd_price
        elif ticker in ['BTC-USD', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'CEMA.L', 'TTE']:  # USD assets
            _, last_price_usd = get_last_price(data, ticker, precision=2)
            value_eur = last_price_usd * quantity / eurusd_price
        elif ticker == 'EURUSD=X':  # Forex, valeur en EUR
            _, last_price = get_last_price(data, ticker, precision=4)
            value_eur = last_price * quantity
        else:  # EUR assets
            _, last_price = get_last_price(data, ticker, precision=2)
            value_eur = last_price * quantity

        asset_values[ticker] = round(value_eur, 2)
        total_value_eur += value_eur

    return {
        "total_value_eur": round(total_value_eur, 2),
        "breakdown": asset_values
    }

def get_portfolio_performance(data, portfolio:dict):
    """
    Calcule la performance du portefeuille sur 1 jour, 1 mois et 1 an en pourcentage.
    Retourne un dict avec les performances.
    """
    _, eurusd_price = get_last_price(data, 'EURUSD=X', precision=4)
    periods = {'1d': 2, '1mo': 2, '1y': 2}
    performance = {}

    for period, precision in periods.items():
        total_now = 0
        total_past = 0
        for ticker, quantity in portfolio.items():
            # Récupère le prix actuel et le prix à la période donnée
            if ticker == 'DBX9.DE':
                _, price_now_eur = get_last_price(data, ticker, precision)
                _, price_past_eur = get_price_at_given_time(data, ticker, period, precision)
                price_now_usd = price_now_eur * eurusd_price
                price_past_usd = price_past_eur * eurusd_price
                value_now_eur = price_now_usd * quantity / eurusd_price
                value_past_eur = price_past_usd * quantity / eurusd_price
            elif ticker in ['BTC-USD', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'CEMA.L', 'TTE']:
                _, price_now_usd = get_last_price(data, ticker, precision)
                _, price_past_usd = get_price_at_given_time(data, ticker, period, precision)
                value_now_eur = price_now_usd * quantity / eurusd_price
                value_past_eur = price_past_usd * quantity / eurusd_price
            elif ticker == 'EURUSD=X':
                _, price_now = get_last_price(data, ticker, precision)
                _, price_past = get_price_at_given_time(data, ticker, period, precision)
                value_now_eur = price_now * quantity
                value_past_eur = price_past * quantity
            else:
                raise ValueError(f"Unsupported ticker for performance calculation: {ticker}")

            total_now += value_now_eur
            total_past += value_past_eur

        if total_past != 0:
            perf = round((total_now - total_past) / total_past * 100, precision)
        else:
            perf = None
        performance[period] = perf

    return {
        "performance_1d_percent": performance['1d'],
        "performance_1mo_percent": performance['1mo'],
        "performance_1y_percent": performance['1y']
    }

if __name__ == "__main__":
    data = load_tickers(['BTC-USD','EURUSD=X', 'GC=F', 'XDW0L.XC', 'HSTE.L', 'DBX9.DE', 'CEMA.L', 'TTE'], interval = '1d', period = '2y')
    _, eurusd_price = get_last_price(data, 'EURUSD=X', precision = 4)
    output = {
        "BTC_USD": get_asset_section(data, 'BTC-USD', precision=2),
        "EUR_USD": get_asset_section(data, 'EURUSD=X', precision=4),
        "GOLD_USD": get_asset_section(data, 'GC=F', precision=2),
        "ENERGY_USD": get_asset_section(data, 'XDW0L.XC', precision=2),
        "TOTAL_ENERGY_USD": get_asset_section(data, 'TTE', precision=2),
        "HKTech_USD": get_asset_section(data, 'HSTE.L', precision=2),
        "ChinaA_USD": get_asset_section(data, 'DBX9.DE', precision=2, conversion_rate=eurusd_price),
        "EmergingMarkets_USD": get_asset_section(data, 'CEMA.L', precision=2)
    }
    print(json.dumps(output, indent=2))
    # portfolio_value = get_portfolio_value_eur(data, PORTFOLIO_DICT)
    # print(json.dumps(portfolio_value, indent=2))
    portfolio_performance = get_portfolio_performance(data, PORTFOLIO_DICT)
    print(json.dumps(portfolio_performance, indent=2))
