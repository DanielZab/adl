"""
This class was not used for the final implementation of the project
"""

import requests
import os

session = requests.Session()

token = os.environ.get("MARKETDATA_API_KEY")
session.headers.update({"Authorization": f"Bearer {token}"})

'''
Required params:
----------------------------------------
resolution: string, The duration of each candle. Only daily candles are supported at this time.
Daily Resolutions: (daily, D, 1D)

symbols: string, The ticker symbols to return in the response, separated by commas. The symbols parameter may be omitted if the snapshot parameter is set to true.
========================================
Optional params:
----------------------------------------
snapshot: boolean, Returns candles for all available symbols for the date indicated. The symbols parameter can be omitted if snapshot is set to true.

date: date, The date of the candles to be returned. If no date is specified, during market hours the candles returned will be from the current session. If the market is closed the candles will be from the most recent session. Accepted date inputs: ISO 8601, unix, spreadsheet.

adjustsplits: boolean, Adjust historical data for for historical splits and reverse splits. Market Data uses the CRSP methodology for adjustment. Daily candles default: true.
'''
def bulk_candles(resolution: str, symbols: list, snapshot: bool | None = None, date: str | None = None, adjustsplits: bool | None = None):
    symbols = ",".join(symbols)
    url = f"https://api.marketdata.app/v1/stocks/bulkcandles/{resolution}/?symbols={symbols}"

    if snapshot:
        url += f"&snapshot={snapshot}"
    if date:
        url += f"&date={date}"
    if adjustsplits:
        url += f"&adjustsplits={adjustsplits}"

    response = session.get(url)

    return response.json()


'''
Required params:
----------------------------------------
symbols: string, The ticker symbols to return in the response, separated by commas. The symbols parameter may be omitted if the snapshot parameter is set to true.
========================================
Optional params:
----------------------------------------
snapshot: boolean, Returns a full market snapshot with quotes for all symbols when set to true. The symbols parameter may be omitted if the snapshot parameter is set.

extended: boolean, Control the inclusion of extended hours data in the quote output. Defaults to true if omitted.
When set to true, the most recent quote is always returned, without regard to whether the market is open for primary trading or extended hours trading.
When set to false, only quotes from the primary trading session are returned. When the market is closed or in extended hours, a historical quote from the last closing bell of the primary trading session is returned instead of an extended hours quote.
'''
def bulk_quotes(symbols: list, snapshot: bool | None = None, extended: bool | None = None):
    symbols = ",".join(symbols)

    url = f"https://api.marketdata.app/v1/stocks/bulkquotes/?symbols={symbols}"

    if snapshot:
        url += f"&snapshot={snapshot}"
    if extended:
        url += f"&extended={extended}"

    response = session.get(url)

    return response.json()


print(bulk_quotes(["GOOG"]))