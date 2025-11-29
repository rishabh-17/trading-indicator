import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import yfinance as yf

st.title("📊 Market CSV Data Downloader")

# Famous markets lists
NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "SUNPHARMA.NS", "HCLTECH.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "POWERGRID.NS", "TECHM.NS", "M&M.NS", "TATAMOTORS.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "ONGC.NS", "WIPRO.NS", "BAJAJFINSV.NS", "NESTLEIND.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS",
    "TATASTEEL.NS", "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS", "BRITANNIA.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "TATACONSUM.NS", "SBILIFE.NS", "APOLLOHOSP.NS",
    "BPCL.NS", "DIVISLAB.NS", "SHRIRAMFIN.NS", "HAVELLS.NS", "BAJAJHLDNG.NS"
]

CRYPTO_POPULAR = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"
]

# Selection inputs
market_type = st.sidebar.selectbox("Market Type", ["Crypto", "Stock"])

if market_type == "Crypto":
    ticker = st.sidebar.selectbox("Symbol", CRYPTO_POPULAR)
else:
    ticker = st.sidebar.selectbox("Symbol", NIFTY50_STOCKS)

timeframe = st.sidebar.selectbox("Candle Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
duration_days = st.sidebar.number_input("Duration (days)", min_value=1, max_value=365, value=30)

# Download CSV
if st.sidebar.button("🚀 Fetch & Download CSV"):
    try:
        now = datetime.utcnow()
        since = now - timedelta(days=duration_days)

        if market_type == "Crypto":
            exchange = ccxt.binance()
            since_ts = exchange.parse8601(since.isoformat())
            limit = 1000
            all_bars = []

            while True:
                bars = exchange.fetch_ohlcv(ticker, timeframe, since=since_ts, limit=limit)
                if not bars:
                    break
                all_bars += bars
                since_ts = bars[-1][0] + 1
                if len(bars) < limit:
                    break

            df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = df["timestamp"].map(lambda x: datetime.utcfromtimestamp(x / 1000))

        else:  # Stock
            data = yf.Ticker(ticker)
            df = data.history(start=since, interval="15m")  # Yahoo supports 15m

            if df.empty:
                st.error("No data returned for this range. Try shorter duration.")
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

        file_name = ticker.replace("/", "_") + "_" + timeframe + "_" + str(duration_days) + "d.csv"
        df.to_csv(file_name, index=False)

        st.success(f"✅ CSV Ready: {file_name} | Rows: {len(df)}")

        st.dataframe(df.head())

        with open(file_name, "rb") as f:
            st.download_button("⬇️ Download CSV", f.read(), file_name=file_name)

    except Exception as e:
        st.error(f"Error: {str(e)}")
