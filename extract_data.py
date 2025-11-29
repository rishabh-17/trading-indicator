import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import yfinance as yf

# Popular crypto & stocks for dropdown
CRYPTO_POPULAR = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT"
]

NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "TATAMOTORS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"
]

# Supported crypto exchanges
EXCHANGES_ALLOWED = ["Kraken"]  # ← Binance & Bybit removed to avoid 451/403 India block

st.title("📊 Market CSV Data Downloader")

# Sidebar selection
market_type = st.sidebar.selectbox("Market Type", ["Crypto", "Stock"])
timeframe = st.sidebar.selectbox("Candle Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
duration_days = st.sidebar.number_input("Duration (days)", min_value=1, max_value=365, value=30)

if market_type == "Crypto":
    symbol = st.sidebar.selectbox("Symbol", CRYPTO_POPULAR)
    exchange = ccxt.kraken()  # ← Will NOT throw 451/403 in India
    use_exchange = True
else:
    symbol = st.sidebar.selectbox("Symbol", NIFTY50_STOCKS)
    use_exchange = False

if st.sidebar.button("🚀 Fetch & Download CSV"):
    try:
        now = datetime.utcnow()
        since_time = now - timedelta(days=duration_days)

        if use_exchange:
            since_ts = exchange.parse8601(since_time.isoformat())
            limit = 1000
            all_bars = []

            while True:
                bars = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
                if not bars:
                    break
                all_bars += bars
                since_ts = bars[-1][0] + 1
                if len(bars) < limit:
                    break

            df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = df["timestamp"].map(lambda x: datetime.utcfromtimestamp(x / 1000))
        
        else:
            data = yf.Ticker(symbol)
            df = data.history(start=since_time, interval="15m")
            if df.empty:
                st.error("No stock data returned. Reduce duration.")
                st.stop()

            df.reset_index(inplace=True)
            df.rename(columns={
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            df["timestamp"] = df["timestamp"].map(lambda x: x.to_pydatetime())

        file_name = symbol.replace("/", "_") + "_" + timeframe + "_" + str(duration_days) + "d.csv"
        df.to_csv(file_name, index=False)

        st.success(f"✅ CSV Ready: {file_name} | Rows: {len(df)}")
        st.dataframe(df.head())

        with open(file_name, "rb") as f:
            st.download_button("⬇️ Download CSV", f.read(), file_name=file_name)

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
