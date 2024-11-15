import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
SYMBOLS = ['SPY', '^NDX', '^GSPC', 'QQQ', 'TQQQ', 'MSFT', 'AAPL',
           'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', '^RUT']
CONSECUTIVE_DAYS=1
def get_current_signals(symbols=SYMBOLS):
    """Get current trading signals and levels for multiple stocks."""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=10)
    all_signals = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start_date, end_date).drop(columns=["Volume", "Adj Close"])
            df['Lower_High'] = df["High"] < df["High"].shift(1)
            df['Higher_Low'] = df["Low"] > df["Low"].shift(1)
            df['Cons_Down'] = df['Lower_High'] * (
                    df['Lower_High'].groupby((df['Lower_High'] != df['Lower_High'].shift()).cumsum()).cumcount() + 1
            )
            df['Cons_Up'] = df['Higher_Low'] * (
                    df['Higher_Low'].groupby((df['Higher_Low'] != df['Higher_Low'].shift()).cumsum()).cumcount() + 1
            )
            latest_data = df.iloc[-1]
            prev_data = df.iloc[-2]
            all_signals[symbol] = {
                'current_price': latest_data['Close'],
                'today_high': latest_data['High'],
                'today_low': latest_data['Low'],
                'prev_high': prev_data['High'],
                'prev_low': prev_data['Low'],
                'cons_down': latest_data['Cons_Down'],
                'cons_up': latest_data['Cons_Up']
            }
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    return all_signals
def print_trading_guide(signals):
    """Print comprehensive trading guide for all symbols."""
    print(f"\n=== Multi-Stock Trading Guide - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    long_setups = []
    short_setups = []
    no_setups = []
    for symbol, data in signals.items():
        if data['cons_down'] >= CONSECUTIVE_DAYS:
            long_setups.append(symbol)
        elif data['cons_up'] >= CONSECUTIVE_DAYS:
            short_setups.append(symbol)
        else:
            no_setups.append(symbol)
    if long_setups:
        print("\n🟢 ACTIVE LONG SETUPS:")
        print("=" * 50)
        for symbol in long_setups:
            data = signals[symbol]
            risk = data['prev_high'] - data['prev_low']
            position_size = round(200 / risk)
            print(f"\n{symbol}:")
            print(f"Current Price: ${data['current_price']:.2f}")
            print(f"LONG Entry Level: Above ${data['prev_high']:.2f}")
            print(f"Risk per share: ${risk:.2f}")
            print(f"Suggested position size: {position_size} shares")
            print(f"Consecutive lower highs: {data['cons_down']:.0f} days")
    if short_setups:
        print("\n🔴 ACTIVE SHORT SETUPS:")
        print("=" * 50)
        for symbol in short_setups:
            data = signals[symbol]
            risk = data['prev_high'] - data['prev_low']
            position_size = round(200 / risk)
            print(f"\n{symbol}:")
            print(f"Current Price: ${data['current_price']:.2f}")
            print(f"SHORT Entry Level: Below ${data['prev_low']:.2f}")
            print(f"Risk per share: ${risk:.2f}")
            print(f"Suggested position size: {position_size} shares")
            print(f"Consecutive higher lows: {data['cons_up']:.0f} days")
    if no_setups:
        print("\n⚪ NO ACTIVE SETUPS:")
        print("=" * 50)
        print(", ".join(no_setups))
def check_earnings():
    """Check upcoming earnings dates for all stocks."""
    print("\nUpcoming Earnings Dates:")
    print("=" * 50)
    for symbol in SYMBOLS:
        if symbol not in ['^NDX', '^GSPC']:
            try:
                stock = yf.Ticker(symbol)
                next_earnings = stock.earnings_dates
                if next_earnings is not None and not next_earnings.empty:
                    next_date = next_earnings.iloc[0]
                    days_until = (next_date - pd.Timestamp.now()).days
                    print(f"{symbol}: {next_date.date()} ({days_until} days away)")
                else:
                    print(f"{symbol}: No earnings date found")
            except:
                print(f"{symbol}: Could not fetch earnings date")
def export_to_csv(signals):
    """Export trading signals and information including entry levels to a CSV file."""
    rows = []
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M')
    for symbol, data in signals.items():
        if data['cons_down'] >= CONSECUTIVE_DAYS:
            setup_type = 'LONG'
            entry_level = f"Above ${data['prev_high']:.2f}"
        elif data['cons_up'] >= CONSECUTIVE_DAYS:
            setup_type = 'SHORT'
            entry_level = f"Below ${data['prev_low']:.2f}"
        else:
            setup_type = 'NONE'
            entry_level = 'No Entry'
        risk = data['prev_high'] - data['prev_low']
        position_size = round(200 / risk) if risk != 0 else 0
        row = {
            'Symbol': symbol,
            'Setup_Type': setup_type,
            'Current_Price': data['current_price'],
            'Entry_Level': entry_level,
            'Previous_High': data['prev_high'],
            'Previous_Low': data['prev_low'],
            'Today_High': data['today_high'],
            'Today_Low': data['today_low'],
            'Risk_Per_Share': risk,
            'Position_Size': position_size,
            'Consecutive_Down_Days': data['cons_down'],
            'Consecutive_Up_Days': data['cons_up'],
            'Date_Generated': dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    filename = f'trading_signals_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\nTrading signals exported to: {filename}")
if __name__ == "__main__":
    print("Fetching current market data and generating signals...")
    signals = get_current_signals()
    print_trading_guide(signals)
    export_to_csv(signals)
    check_earnings()