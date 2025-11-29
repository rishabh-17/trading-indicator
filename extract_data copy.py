import ccxt
import pandas as pd
from datetime import datetime, timedelta

exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "15m"

# Start of last 365 days
now = datetime.utcnow()
start_of_year = now - timedelta(days=365)
since = exchange.parse8601(start_of_year.isoformat())
limit = 1000  # max candles per API call

all_bars = []

while True:
    bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not bars:
        break
    
    all_bars += bars
    since = bars[-1][0] + 1  # move cursor forward
    
    if len(bars) < limit:  # exit if last batch
        break

df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = df["timestamp"].map(lambda x: datetime.utcfromtimestamp(x / 1000))

file_name = "btc_usd_15m_whole_year.csv"
df.to_csv(file_name, index=False)

print("Saved:", file_name, "| Rows:", len(df))
