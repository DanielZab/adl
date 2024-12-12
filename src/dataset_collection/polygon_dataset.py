import requests, os, time, json

SLEEP_TIME = 60 / 4

session = requests.Session()

token = os.environ.get("POLYGON_API_KEY")
session.headers.update({"Authorization": f"Bearer {token}"})


DEFAULT_START_DATE = "2022-01-12"
DEFAULT_END_DATE = "2024-11-25"
DEFAULT_MULTIPLIER = "15"
DEFAULT_TIMESPAN = "minute"
ticker = "AAPL"



def store_stock(ticker: str, multiplier: str = DEFAULT_MULTIPLIER, timespan: str = DEFAULT_TIMESPAN, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE):
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

store_stock("ZM")
store_stock("SPOT")
store_stock("NVDA")
store_stock("AMD")
store_stock("INTC")
store_stock("RWE")