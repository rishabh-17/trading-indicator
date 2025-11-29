import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

# ------ Indicator Calculations ------

def add_indicators(df):
    df = df.copy()

    # RSI 14
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()

    # MACD
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # EMA 20/50
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()

    # Bollinger 20,2
    bb = BollingerBands(close=df["close"])
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # Supertrend
    df = add_supertrend(df)
    return df

def add_supertrend(df, period=10, multiplier=3):
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high-low,
        (high-close.shift()).abs(),
        (low-close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high+low)/2
    ub, lb = hl2+multiplier*atr, hl2-multiplier*atr
    st_line = np.where(close > ub.shift(), lb, np.where(close < lb.shift(), ub, np.nan))
    direction = np.where(close > ub.shift(), 1, np.where(close < lb.shift(), -1, 0))
    df["supertrend"] = st_line
    df["st_dir"] = direction
    return df

# ------ Individual Indicator Signal Logic ------

def indicator_signal(df, name):
    """Return series 1 for LONG, -1 for SHORT, 0 for none"""
    sig = pd.Series(0, index=df.index)

    if name == "RSI":
        rsi_prev = df["rsi"].shift()
        sig[(rsi_prev < 30) & (df["rsi"] >= 30)] = 1
        sig[(rsi_prev > 70) & (df["rsi"] <= 70)] = -1

    elif name == "MACD":
        sig[(df["macd"].shift() < df["macd_signal"].shift()) & (df["macd"] > df["macd_signal"])] = 1
        sig[(df["macd"].shift() > df["macd_signal"].shift()) & (df["macd"] < df["macd_signal"])] = -1

    elif name == "EMA 20/50":
        sig[(df["ema_20"].shift() < df["ema_50"].shift()) & (df["ema_20"] > df["ema_50"])] = 1
        sig[(df["ema_20"].shift() > df["ema_50"].shift()) & (df["ema_20"] < df["ema_50"])] = -1

    elif name == "Bollinger":
        sig[df["close"].shift() < df["bb_lower"].shift()] = 1
        sig[df["close"].shift() > df["bb_upper"].shift()] = -1

    elif name == "Supertrend":
        sig[(df["st_dir"].shift() == -1) & (df["st_dir"] == 1)] = 1
        sig[(df["st_dir"].shift() == 1) & (df["st_dir"] == -1)] = -1

    return sig

# ------ Backtest Engine with Confluence ------

def backtest_confluence(df, indicators, capital=100000, tp=0.02, sl=0.005):
    capital_init = capital
    trades = []
    i = 1

    while i < len(df)-2:
        confluence_signal = None
        price = df["close"].iloc[i]
        time = df["timestamp"].iloc[i]

        # Check if all selected indicators agree
        sigs = [indicator_signal(df, ind).iloc[i] for ind in indicators]
        if len(set(sigs)) == 1 and sigs[0] != 0:
            confluence_signal = sigs[0]

        if confluence_signal is None:
            i += 1
            continue

        direction = confluence_signal
        qty = capital/price
        tp_price = price*(1+tp) if direction==1 else price*(1-tp)
        sl_price = price*(1-sl) if direction==1 else price*(1+sl)
        hit = False

        for j in range(i+1, len(df)):
            high, low = df["high"].iloc[j], df["low"].iloc[j]
            if direction==1:
                if low<=sl_price or high>=tp_price:
                    exit_p = sl_price if low<=sl_price else tp_price
                    reason = "SL" if exit_p==sl_price else "TP"
                    pnl = (exit_p-price)*qty
                    capital += pnl
                    trades.append({"Entry":time,"Exit":df["timestamp"].iloc[j],
                                   "Type":"BUY","Price":price,"Exit Price":exit_p,
                                   "PnL":pnl,"Balance":capital,"Reason":reason})
                    i = j+1
                    hit = True
                    break
            else:
                if high>=sl_price or low<=tp_price:
                    exit_p = sl_price if high>=sl_price else tp_price
                    reason = "SL" if exit_p==sl_price else "TP"
                    pnl = (price-exit_p)*qty
                    capital += pnl
                    trades.append({"Entry":time,"Exit":df["timestamp"].iloc[j],
                                   "Type":"SELL","Price":price,"Exit Price":exit_p,
                                   "PnL":pnl,"Balance":capital,"Reason":reason})
                    i = j+1
                    hit = True
                    break
        if not hit: i += 1

    trades_df = pd.DataFrame(trades)
    return trades_df, capital_init, capital

# ------ Streamlit UI ------

def main():
    st.set_page_config("BTC Backtest (Confluence)", layout="wide")
    st.title("BTC/USD Multi-Indicator Confluence Backtester")

    file = st.sidebar.file_uploader("Upload OHLCV CSV",["csv"])
    cap = st.sidebar.number_input("Capital", 1000.0, 100000.0, 100000.0)


    # Multi indicator selector
    inds = st.sidebar.multiselect("Select Indicators for Confluence",
                                  ["RSI","MACD","EMA 20/50","Bollinger","Supertrend"],
                                  ["RSI","Supertrend"])

    tp = 2.0/100
    sl = 0.5/100

    if file:
        df = pd.read_csv(file)
        df.columns=[c.lower() for c in df.columns]
        df["timestamp"]=pd.to_datetime(df["timestamp"])
        df=df.sort_values("timestamp").reset_index(drop=True)
        df=add_indicators(df)
        df["signal"]=0

        for ind in inds:
            df["signal"] += indicator_signal(df, ind)

        # Normalize aggregated signal to -1,0,1
        df["signal"] = np.sign(df["signal"])

        tdf,bal_init,bal_final = backtest_confluence(df, inds, cap, tp, sl)

        st.subheader("Trade Log")
        st.dataframe(tdf)

        st.subheader("Balance Summary")
        s1,s2,s3 = st.columns(3)
        s1.metric("Initial Balance", f"{bal_init:,.2f}")
        s2.metric("Final Balance", f"{bal_final:,.2f}")
        s3.metric("Total PnL", f"{bal_final-bal_init:,.2f}")

        st.subheader("Price Chart with Trade Markers")
        fig=go.Figure(go.Candlestick(x=df.timestamp,open=df.open,high=df.high,
                                    low=df.low,close=df.close))
        if not tdf.empty:
            buys=tdf[tdf.Type=="BUY"]; sells=tdf[tdf.Type=="SELL"]
            fig.add_trace(go.Scatter(x=buys.Entry,y=buys.Price,mode="markers",
                                     name="BUY",marker_symbol="triangle-up",marker_size=10))
            fig.add_trace(go.Scatter(x=sells.Entry,y=sells.Price,mode="markers",
                                     name="SELL",marker_symbol="triangle-down",marker_size=10))
        fig.update_layout(height=600,xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Equity Curve")
        eq=go.Figure(go.Scatter(x=tdf.Exit,y=tdf.Balance,mode="lines+markers"))
        eq.update_layout(height=400)
        st.plotly_chart(eq,use_container_width=True)

    else:
        st.info("Upload a CSV to begin")

if __name__ == "__main__":
    main()
