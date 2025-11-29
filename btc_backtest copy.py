import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands


# -------------------- Indicator Calculations -------------------- #

def add_indicators(df):
    df = df.copy()

    # RSI
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    # MACD
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # EMA
    ema_fast = EMAIndicator(close=df["close"], window=20)
    ema_slow = EMAIndicator(close=df["close"], window=50)
    df["ema_20"] = ema_fast.ema_indicator()
    df["ema_50"] = ema_slow.ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # Supertrend
    df = add_supertrend(df, period=10, multiplier=3)

    return df


def add_supertrend(df, period=10, multiplier=3):
    """
    Basic Supertrend implementation.
    Adds columns: 'supertrend', 'supertrend_direction'
    direction: 1 = bullish, -1 = bearish
    """
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR
    atr = tr.rolling(period).mean()

    # Basic Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_ub = upper_band.copy()
    final_lb = lower_band.copy()

    for i in range(1, len(df)):
        if upper_band.iloc[i] < final_ub.iloc[i - 1] or close.iloc[i - 1] > final_ub.iloc[i - 1]:
            final_ub.iloc[i] = upper_band.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        if lower_band.iloc[i] > final_lb.iloc[i - 1] or close.iloc[i - 1] < final_lb.iloc[i - 1]:
            final_lb.iloc[i] = lower_band.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

    supertrend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int")

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = np.nan
            direction.iloc[i] = 1
            continue

        if supertrend.iloc[i - 1] == final_ub.iloc[i - 1]:
            if close.iloc[i] <= final_ub.iloc[i]:
                supertrend.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == final_lb.iloc[i - 1]:
            if close.iloc[i] >= final_lb.iloc[i]:
                supertrend.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1
        else:
            if close.iloc[i] > final_lb.iloc[i]:
                supertrend.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1

    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction

    return df


# -------------------- Signal Generators -------------------- #

def generate_signals(df, strategy_name):
    df = df.copy()
    df["signal"] = 0  # 1 = long, -1 = short, 0 = no trade

    if strategy_name == "RSI":
        # Long when RSI crosses above 30, Short when crosses below 70
        rsi_prev = df["rsi"].shift(1)
        df.loc[(rsi_prev < 30) & (df["rsi"] >= 30), "signal"] = 1
        df.loc[(rsi_prev > 70) & (df["rsi"] <= 70), "signal"] = -1

    elif strategy_name == "MACD":
        macd_prev = df["macd"].shift(1)
        signal_prev = df["macd_signal"].shift(1)
        # MACD bullish cross
        df.loc[(macd_prev < signal_prev) & (df["macd"] > df["macd_signal"]), "signal"] = 1
        # MACD bearish cross
        df.loc[(macd_prev > signal_prev) & (df["macd"] < df["macd_signal"]), "signal"] = -1

    elif strategy_name == "EMA Crossover (20/50)":
        ema20_prev = df["ema_20"].shift(1)
        ema50_prev = df["ema_50"].shift(1)
        # Golden cross
        df.loc[(ema20_prev < ema50_prev) & (df["ema_20"] > df["ema_50"]), "signal"] = 1
        # Death cross
        df.loc[(ema20_prev > ema50_prev) & (df["ema_20"] < df["ema_50"]), "signal"] = -1

    elif strategy_name == "Bollinger Bounce":
        close_prev = df["close"].shift(1)
        lower_prev = df["bb_lower"].shift(1)
        upper_prev = df["bb_upper"].shift(1)

        # Price bouncing up from lower band
        df.loc[(close_prev < lower_prev) & (df["close"] > df["bb_lower"]), "signal"] = 1
        # Price bouncing down from upper band
        df.loc[(close_prev > upper_prev) & (df["close"] < df["bb_upper"]), "signal"] = -1

    elif strategy_name == "Supertrend":
        dir_prev = df["supertrend_direction"].shift(1)
        dir_now = df["supertrend_direction"]
        # Direction change
        df.loc[(dir_prev == -1) & (dir_now == 1), "signal"] = 1
        df.loc[(dir_prev == 1) & (dir_now == -1), "signal"] = -1

    return df


# -------------------- Backtest Engine -------------------- #

def simulate_trade(df, start_index, direction, entry_price, tp_pct=0.02, sl_pct=0.005):
    """
    direction: 1 = long, -1 = short
    Returns: exit_index, exit_price, reason
    """
    if start_index >= len(df):
        return None, None, "no_data"

    if direction == 1:
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    for i in range(start_index, len(df)):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # Both TP and SL in same candle - we take worst case (SL first)
        if direction == 1:
            if low <= sl_price and high >= tp_price:
                return i, sl_price, "SL (both-hit)"
            if low <= sl_price:
                return i, sl_price, "SL"
            if high >= tp_price:
                return i, tp_price, "TP"
        else:
            if high >= sl_price and low <= tp_price:
                return i, sl_price, "SL (both-hit)"
            if high >= sl_price:
                return i, sl_price, "SL"
            if low <= tp_price:
                return i, tp_price, "TP"

    # If never hit, exit at last close
    return len(df) - 1, df["close"].iloc[-1], "Exit@End"


def backtest(df, initial_capital=100000, tp_pct=0.02, sl_pct=0.005):
    trades = []
    capital = initial_capital

    i = 1  # start from 1 because many indicators need previous values
    n = len(df)

    while i < n - 1:
        signal = df["signal"].iloc[i]

        if signal == 0:
            i += 1
            continue

        direction = signal  # 1 long, -1 short
        entry_price = df["close"].iloc[i]
        entry_time = df["timestamp"].iloc[i]

        if entry_price <= 0:
            i += 1
            continue

        # All-in each trade with current capital
        qty = capital / entry_price

        exit_index, exit_price, reason = simulate_trade(
            df, i + 1, direction, entry_price, tp_pct, sl_pct
        )

        if exit_index is None:
            break

        exit_time = df["timestamp"].iloc[exit_index]
        pnl = (exit_price - entry_price) * qty * direction
        capital += pnl

        trades.append(
            {
                "entry_index": i,
                "exit_index": exit_index,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "reason": reason,
                "qty": qty,
                "pnl": pnl,
                "balance_after": capital,
            }
        )

        # Continue after exit candle
        i = exit_index + 1

    trades_df = pd.DataFrame(trades)
    return trades_df, capital


def compute_stats(trades_df, initial_capital):
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "final_balance": initial_capital,
            "max_drawdown": 0.0,
        }

    total_trades = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum()
    win_rate = wins / total_trades * 100

    total_pnl = trades_df["pnl"].sum()
    final_balance = initial_capital + total_pnl

    balance_curve = trades_df["balance_after"]
    peak = balance_curve.cummax()
    drawdown = (balance_curve - peak) / peak
    max_dd = drawdown.min() * 100  # in %

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "final_balance": final_balance,
        "max_drawdown": max_dd,
    }


# -------------------- Streamlit UI -------------------- #

def main():
    st.set_page_config(page_title="BTC/USD Indicator Backtester", layout="wide")
    st.title("📈 BTC/USD Indicator Backtester")

    st.sidebar.header("Backtest Settings")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (timestamp,open,high,low,close,volume)", type=["csv"]
    )

    initial_capital = st.sidebar.number_input(
        "Initial Capital", value=100000.0, step=1000.0
    )

    tp_pct = st.sidebar.number_input(
        "Take Profit %", value=2.0, step=0.1
    ) / 100.0

    sl_pct = st.sidebar.number_input(
        "Stoploss %", value=0.5, step=0.1
    ) / 100.0

    strategy_name = st.sidebar.selectbox(
        "Select Indicator Strategy",
        [
            "RSI",
            "MACD",
            "EMA Crossover (20/50)",
            "Bollinger Bounce",
            "Supertrend",
        ],
    )

    if uploaded_file is None:
        st.info("👆 Upload your BTC/USD OHLCV CSV to start backtesting.")
        return

    # Load data
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"CSV must contain columns: {required_cols}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add indicators
    df = add_indicators(df)

    # Generate signals
    df = generate_signals(df, strategy_name)

    # Run backtest
    trades_df, final_balance = backtest(df, initial_capital, tp_pct, sl_pct)
    stats = compute_stats(trades_df, initial_capital)

    st.subheader("Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Initial Balance", f"{initial_capital:,.2f}")
    col2.metric("Final Balance", f"{stats['final_balance']:,.2f}")
    col3.metric("Total PnL", f"{stats['total_pnl']:,.2f}")
    col4.metric("Win Rate", f"{stats['win_rate']:.2f}%")
    col5.metric("Max Drawdown", f"{stats['max_drawdown']:.2f}%")

    st.subheader("Trades Table")
    if trades_df.empty:
        st.warning("No trades generated by this strategy / data.")
    else:
        st.dataframe(trades_df, use_container_width=True)

    # Charts
    st.subheader("Price Chart with Trades")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            )
        ]
    )

    if not trades_df.empty:
        # Entry markers
        long_entries = trades_df[trades_df["direction"] == "LONG"]
        short_entries = trades_df[trades_df["direction"] == "SHORT"]

        fig.add_trace(
            go.Scatter(
                x=long_entries["entry_time"],
                y=long_entries["entry_price"],
                mode="markers",
                name="Long Entry",
                marker_symbol="triangle-up",
                marker_size=10,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=short_entries["entry_time"],
                y=short_entries["entry_price"],
                mode="markers",
                name="Short Entry",
                marker_symbol="triangle-down",
                marker_size=10,
            )
        )

    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Equity curve
    st.subheader("Equity Curve (Balance After Each Trade)")
    if not trades_df.empty:
        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(
                x=trades_df["exit_time"],
                y=trades_df["balance_after"],
                mode="lines+markers",
                name="Equity",
            )
        )
        eq_fig.update_layout(height=400)
        st.plotly_chart(eq_fig, use_container_width=True)


if __name__ == "__main__":
    main()
