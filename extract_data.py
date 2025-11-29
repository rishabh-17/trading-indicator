import ccxt
import pandas as pd
from datetime import datetime, timedelta

exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "15m"

# Calculate start of last full month
now = datetime.utcnow()
first_of_current_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
start_of_last_month = first_of_current_month - timedelta(days=1)
start_of_last_month = start_of_last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

since = exchange.parse8601(start_of_last_month.isoformat())
limit = 1000

all_bars = []
while True:
    bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not bars:
        break
    all_bars += bars
    since = bars[-1][0] + 1
    if len(bars) < limit:
        break

df = pd.DataFrame(all_bars, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = df["timestamp"].map(lambda x: datetime.utcfromtimestamp(x/1000))

file_name = "btc_usd_15m_whole_month.csv"
df.to_csv(file_name, index=False)
print("Saved:", file_name, "| Rows:", len(df))
