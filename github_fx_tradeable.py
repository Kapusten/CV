import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
def get_current_signals(symbol='EURUSD=X'):
    """Get current trading signals and levels."""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=10)
    df = yf.download(symbol, start_date, end_date).drop(columns=["Volume", "Adj Close"])
    df['Higher_High'] = df["High"] > df["High"].shift(1)
    df['Lower_Low'] = df["Low"] < df["Low"].shift(1)
    df['Consecutive_Higher_Highs'] = df['Higher_High'] * (
            df['Higher_High'].groupby((df['Higher_High'] != df['Higher_High'].shift()).cumsum()).cumcount() + 1
    )
    df['Consecutive_Lower_Lows'] = df['Lower_Low'] * (
            df['Lower_Low'].groupby((df['Lower_Low'] != df['Lower_Low'].shift()).cumsum()).cumcount() + 1
    )
    latest_data = df.iloc[-1]
    prev_data = df.iloc[-2]
    print(f"\n=== Trading Guide for {symbol} - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    print("Current Price Levels:")
    print(f"Current Price: {latest_data['Close']:.4f}")
    print(f"Today's High: {latest_data['High']:.4f}")
    print(f"Today's Low: {latest_data['Low']:.4f}")
    print("\nKey Reference Levels:")
    print(f"Previous Day's High: {prev_data['High']:.4f}")
    print(f"Previous Day's Low: {prev_data['Low']:.4f}")
    print("\nPattern Analysis:")
    print(f"Consecutive Higher Highs: {latest_data['Consecutive_Higher_Highs']:.0f}")
    print(f"Consecutive Lower Lows: {latest_data['Consecutive_Lower_Lows']:.0f}")
    print("\nTrading Signals:")
    if latest_data['Consecutive_Lower_Lows'] >= 1:
        print("\n🟢 LONG SETUP ACTIVE:")
        print(f"Entry Level: Below {prev_data['Low']:.4f}")
        print("Trading Plan:")
        print("1. Wait for price to break below previous day's low")
        print("2. Enter LONG position when break occurs")
        print("3. Use market order if gap down occurs at open")
        print("4. Use limit order near the break level during the day")
    if latest_data['Consecutive_Higher_Highs'] >= 1:
        print("\n🔴 SHORT SETUP ACTIVE:")
        print(f"Entry Level: Above {prev_data['High']:.4f}")
        print("Trading Plan:")
        print("1. Wait for price to break above previous day's high")
        print("2. Enter SHORT position when break occurs")
        print("3. Use market order if gap up occurs at open")
        print("4. Use limit order near the break level during the day")
    if (latest_data['Consecutive_Higher_Highs'] < 1 and
            latest_data['Consecutive_Lower_Lows'] < 1):
        print("\n⚪ NO ACTIVE SETUP")
        print("Wait for new pattern to develop")
if __name__ == "__main__":
    get_current_signals()