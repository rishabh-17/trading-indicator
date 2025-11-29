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

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
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
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2

    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr

    df["supertrend"] = np.where(close > ub.shift(), lb, np.where(close < lb.shift(), ub, np.nan))
    df["st_dir"] = np.where(close > ub.shift(), 1, np.where(close < lb.shift(), -1, 0))

    return df

# ------ Individual Indicator Signal Logic ------

def indicator_signal(df, name):
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
        sig[df["close"] < df["bb_lower"]] = 1
        sig[df["close"] > df["bb_upper"]] = -1

    elif name == "Supertrend":
        sig[(df["st_dir"].shift() == -1) & (df["st_dir"] == 1)] = 1
        sig[(df["st_dir"].shift() == 1) & (df["st_dir"] == -1)] = -1

    return sig

# ------ Confluence Backtest Engine ------

def backtest_confluence(df, indicators, capital, tp, sl):
    cap = capital  # work on local copy
    trades = []
    i = 1

    while i < len(df) - 1:
        price = df["close"].iloc[i]
        time = df["timestamp"].iloc[i]

        # Check confluence
        signals = [indicator_signal(df, ind).iloc[i] for ind in indicators]
        if len(set(signals)) != 1 or signals[0] == 0:
            i += 1
            continue

        direction = signals[0]
        qty = cap / price

        tp_price = price * (1 + tp) if direction == 1 else price * (1 - tp)
        sl_price = price * (1 - sl) if direction == 1 else price * (1 + sl)

        hit = False

        for j in range(i + 1, len(df)):
            high, low = df["high"].iloc[j], df["low"].iloc[j]

            if direction == 1:  # BUY (LONG)
                if high >= tp_price:
                    exit_price, reason = tp_price, "TP"
                    hit = True
                elif low <= sl_price:
                    exit_price, reason = sl_price, "SL"
                    hit = True
            else:  # SELL (SHORT)
                if low <= tp_price:
                    exit_price, reason = tp_price, "TP"
                    hit = True
                elif high >= sl_price:
                    exit_price, reason = sl_price, "SL"
                    hit = True
                if hit: exit_price = sl_price if exit_price == sl_price else tp_price

            if hit:
                pnl = (exit_price - price) * qty if direction == 1 else (price - exit_price) * qty
                cap += pnl
                trades.append({
                    "Entry Time": time,
                    "Exit Time": df["timestamp"].iloc[j],
                    "Trade Type": "LONG" if direction == 1 else "SHORT",
                    "Entry Price": price,
                    "Exit Price": exit_price,
                    "Reason": reason,
                    "Quantity": qty,
                    "PnL": round(pnl, 2),
                    "Balance After Trade": round(cap, 2)
                })
                i = j
                break

        i += 1

    trades_df = pd.DataFrame(trades)
    return trades_df, capital, cap


# ------ Streamlit UI ------

def main():
    st.set_page_config("BTC Backtest", layout="wide")
    st.title("📊 BTC/USD Multi-Indicator Confluence Backtester")

    file = st.sidebar.file_uploader("Upload OHLCV CSV", ["csv"])

    capital_input = st.sidebar.number_input("Initial Capital", min_value=1000.0, value=100000.0, step=1000.0)

    indicators = st.sidebar.multiselect("Select Indicators for Confluence",
                                        ["RSI", "MACD", "EMA 20/50", "Bollinger", "Supertrend"],
                                        [])

    target = st.sidebar.number_input("Target %", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100
    stoploss = st.sidebar.number_input("Stoploss %", min_value=0.1, max_value=5.0, value=0.5, step=0.1) / 100

    # Prevent auto-run using session state
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False

    if file:
        df = pd.read_csv(file)
        df.columns = [c.lower().strip() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = add_indicators(df)
        df["signal"] = 0  # reset

        st.session_state.run_analysis = st.sidebar.button("🚀 Start Analysis")

        if not indicators:
            st.warning("Select at least 1 indicator for confluence.")
            return

        if st.session_state.run_analysis:
            # Create signal when all indicators agree
            signal_series = sum([indicator_signal(df, ind) for ind in indicators])
            df["signal"] = np.sign(signal_series)

            trades_df, capital_start, capital_end = backtest_confluence(df, indicators, capital_input, target, stoploss)

            st.subheader("Trade Log")
            if trades_df.empty:
                st.warning("No matching trades found!")
            else:
                st.dataframe(trades_df, use_container_width=True)

            st.subheader("Balance Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"{capital_start:,.2f}")
            col2.metric("Final Balance", f"{capital_end:,.2f}")
            col3.metric("Net PnL", f"{capital_end - capital_start:,.2f}")

            st.subheader("Candlestick Chart")
            fig = go.Figure(go.Candlestick(
                x=df.timestamp, open=df.open, high=df.high, low=df.low, close=df.close
            ))
            if not trades_df.empty:
                long = trades_df[trades_df["Trade Type"] == "LONG"]
                short = trades_df[trades_df["Trade Type"] == "SHORT"]
                fig.add_trace(go.Scatter(x=long["Entry Time"], y=long["Entry Price"], mode="markers",
                                         name="LONG", marker_symbol="triangle-up", marker_size=12))
                fig.add_trace(go.Scatter(x=short["Entry Time"], y=short["Entry Price"], mode="markers",
                                         name="SHORT", marker_symbol="triangle-down", marker_size=12))
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Equity Curve")
            eq = go.Figure(go.Scatter(
                x=trades_df["Exit Time"],
                y=trades_df["Balance After Trade"],
                mode="lines+markers",
                name="Equity Curve"
            ))
            eq.update_layout(height=400)
            st.plotly_chart(eq, use_container_width=True)

        else:
            st.info("Click **Start Analysis** to run backtest.")

    else:
        st.info("Upload CSV to begin")

if __name__ == "__main__":
    main()
