'''
This script is used to get the stock data as json files for selected tickers and store it in a folder with the ticker name. 
The least recent data that can be fetched in the free tier is today - 2 years. There is also a rate limit for the API, due to monetary constraints.

The idea was to include stocks with different patterns, such as a steady growth (NVDA), a volatile nature (INTC), and a decline (ZM).

The dataset "RWE" seems to be faulty, which is why it was not included in the final dataset.
'''

import json
import os
import requests
import time

# Sleep timer to avoid hitting the API rate limit
SLEEP_TIME = 60 / 4

session = requests.Session()

token = os.environ.get("POLYGON_API_KEY")
session.headers.update({"Authorization": f"Bearer {token}"})

DEFAULT_START_DATE = "2022-01-12"
DEFAULT_END_DATE = "2024-11-25"
DEFAULT_MULTIPLIER = "15"
DEFAULT_TIMESPAN = "minute"


def store_stock(ticker: str, multiplier: str = DEFAULT_MULTIPLIER, timespan: str = DEFAULT_TIMESPAN,
                start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE):
    '''
    Gets the stock data for the given ticker as candles and stores it in a folder with the ticker name

    ticker: The ticker of the stock
    multiplier: The multiplier for the stock data to quantify how many of the timespan units are in one candle
    timespan: The timespan for the stock data such as day, second, minute, hour...
    start_date: The start date for the stock data
    end_date: The end date for the stock data
    '''
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc"

    urls = [url]
    os.mkdir(ticker)
    counter = 0
    while len(urls) > 0:
        url = urls.pop(0)
        response = session.get(url)
        fail_cnt = 0
        while response.status_code != 200:
            time.sleep(SLEEP_TIME)
            response = session.get(url)
            fail_cnt += 1
            if fail_cnt > 10:
                return

        data = response.json()
        with open(f"{ticker}\\{counter}.json", "w") as f:
            json.dump(data, f)

        counter += 1
        if "next_url" in data:
            urls.append(data["next_url"])
        time.sleep(SLEEP_TIME)


store_stock("AAPL")
store_stock("ZM")
store_stock("SPOT")
store_stock("NVDA")
store_stock("AMD")
store_stock("INTC")
store_stock("RWE")
