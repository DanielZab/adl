
import time

def get_stock_price(ticker_symbol):
    """Scrape the current stock price for the given ticker symbol from Yahoo Finance."""
    url = f"https://finance.yahoo.com/quote/{ticker_symbol}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to fetch data from Yahoo Finance.")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    #price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
    price_tag = soup.find('span', {'data-testid': 'qsp-price'})


    if price_tag:
        return float(price_tag.text.replace(',', ''))

    print("Failed to parse stock price.")
    return None

def monitor_stock_price(ticker_symbol, interval=300):
    """Monitor the stock price and print it every specified interval."""
    print(f"Monitoring stock price for {ticker_symbol} every {interval // 60} minutes...")

    while True:
        current_price = get_stock_price(ticker_symbol)

        if current_price is not None:
            print(f"Stock price for {ticker_symbol}: ${current_price:.2f}")

        time.sleep(interval)

if __name__ == "__main__":
    # Replace 'AAPL' with the desired stock ticker symbol
    ticker_symbol = "NVDA"
    # Interval in seconds to check for price updates (300 seconds = 5 minutes)
    interval = 300
    monitor_stock_price(ticker_symbol, interval)