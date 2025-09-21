import yfinance as yf
import pandas as pd
import numpy as np
import pytz, json

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

if __name__ == "__main__":
    data = load_tickers(['BTC-USD','EURUSD=X', 'GC=F'], interval = '1d', period = '2y')
    # print(data.tail(20))
    _, btc_price = get_last_price(data, 'BTC-USD', precision = 2)
    btc_change_1d = get_price_evolution(data, 'BTC-USD', '1d', precision = 2)
    btc_change_1mo = get_price_evolution(data, 'BTC-USD', '1mo', precision = 2)
    btc_change_1y = get_price_evolution(data, 'BTC-USD', '1y', precision = 2)
    # EURUSD
    _, eurusd_price = get_last_price(data, 'EURUSD=X', precision = 4)
    eurusd_change_1d = get_price_evolution(data, 'EURUSD=X', '1d', precision = 4)
    eurusd_change_1mo = get_price_evolution(data, 'EURUSD=X', '1mo', precision = 4)
    eurusd_change_1y = get_price_evolution(data, 'EURUSD=X', '1y', precision = 4)
    # GOLD
    _, gold_price = get_last_price(data, 'GC=F', precision = 2)
    gold_change_1d = get_price_evolution(data, 'GC=F', '1d', precision = 2)
    gold_change_1mo = get_price_evolution(data, 'GC=F', '1mo', precision = 2)
    gold_change_1y = get_price_evolution(data, 'GC=F', '1y', precision = 2)

    output = {
        "BTC-USD": {
            "last_price": btc_price,
            "change_1d_percent": btc_change_1d,
            "change_1mo_percent": btc_change_1mo,
            "change_1y_percent": btc_change_1y
        },
        "EURUSD=X": {
            "last_price": eurusd_price,
            "change_1d_percent": eurusd_change_1d,
            "change_1mo_percent": eurusd_change_1mo,
            "change_1y_percent": eurusd_change_1y
        },
        "GC=F": {
            "last_price": gold_price,
            "change_1d_percent": gold_change_1d,
            "change_1mo_percent": gold_change_1mo,
            "change_1y_percent": gold_change_1y
        }
    }
    print(json.dumps(output, indent=2))
