import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
START_DATE = dt.datetime(2010, 1, 1)
END_DATE = dt.datetime(2024, 11, 4)
SYMBOLS = ['EURPLN=X']
START_BALANCE = 10000
CONSECUTIVE_DAYS = 1
def get_price_data(symbol, start=START_DATE, end=END_DATE):
    """Download and clean historical stock data."""
    df = yf.download(symbol, start, end).drop(columns=["Volume", "Adj Close"])
    df["Close"] = df["Open"].shift(-1)
    df = df.dropna()
    return df
def calculate_indicators(df):
    """Calculate daily indicators and consecutive highs/lows."""
    df["Close_to_Open_Ratio"] = df["Close"] / df["Open"]
    df["Close_to_PrevHigh_Ratio"] = df["Close"] / df["High"].shift(1)
    df["Close_to_PrevLow_Ratio"] = df["Close"] / df["Low"].shift(1)
    df['Higher_High'] = df["High"] > df["High"].shift(1)
    df['Lower_Low'] = df["Low"] < df["Low"].shift(1)
    df['Consecutive_Higher_Highs'] = df['Higher_High'] * (
            df['Higher_High'].groupby((df['Higher_High'] != df['Higher_High'].shift()).cumsum()).cumcount() + 1
    )
    df['Consecutive_Lower_Lows'] = df['Lower_Low'] * (
            df['Lower_Low'].groupby((df['Lower_Low'] != df['Lower_Low'].shift()).cumsum()).cumcount() + 1
    )
    return df
def generate_signals(df, consecutive_days=CONSECUTIVE_DAYS):
    """Generate trading signals based on price patterns."""
    df["Long_Signal"] = df['Consecutive_Lower_Lows'] >= consecutive_days
    df["Short_Signal"] = df['Consecutive_Higher_Highs'] >= consecutive_days
    df["Previous_High_Broken"] = (df["Open"] > df["High"].shift(1)) | (df["High"] >= df["High"].shift(1))
    df["Previous_Low_Broken"] = (df["Open"] < df["Low"].shift(1)) | (df["Low"] <= df["Low"].shift(1))
    df["Long_Trade"] = df["Long_Signal"].shift(1) & df["Previous_Low_Broken"]
    df["Short_Trade"] = df["Short_Signal"].shift(1) & df["Previous_High_Broken"]
    df['Long_Return'] = np.where(
        df["Long_Trade"],
        np.where(df["Open"] < df["Low"].shift(1),
                 df["Close_to_Open_Ratio"],
                 df["Close_to_PrevLow_Ratio"]),
        1
    )
    df['Short_Return'] = np.where(
        df["Short_Trade"] & ~df["Long_Trade"],
        np.where(df["Open"] > df["High"].shift(1),
                 2 - df["Close_to_Open_Ratio"],
                 2 - df["Close_to_PrevHigh_Ratio"]),
        1
    )
    return df
def plot_balance(results):
    """Plot the balance over time."""
    plt.style.use("dark_background")
    plt.figure(figsize=(16, 8))
    plt.rcParams.update({"font.size": 18})
    plt.plot(results["Balance"])
    plt.title("Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.show()
def process_symbols(symbols=SYMBOLS):
    """Process all symbols and generate signals."""
    return [generate_signals(calculate_indicators(get_price_data(symbol))) for symbol in symbols]
def simulate_trading(dfs, start_balance=START_BALANCE, start_date=START_DATE, end_date=END_DATE):
    """Simulate trading with the generated signals."""
    balances = [start_balance]
    dates = [start_date]
    in_market = [False]
    day = start_date
    while day < end_date:
        trades, sum_ret = 0, 0
        day += dt.timedelta(days=1)
        for df in dfs:
            if day in df.index:
                data = df.loc[day]
                if data["Long_Trade"] or data["Short_Trade"]:
                    trades += 1
                    sum_ret += data["Long_Return"] * data["Short_Return"]
        combined_ret = sum_ret / trades if trades > 0 else 1
        new_balance = balances[-1] * combined_ret
        balances.append(new_balance)
        dates.append(day)
        in_market.append(trades > 0)
    return pd.DataFrame({
        "Date": dates,
        "Balance": balances,
        "In_Market": in_market
    }).set_index("Date")
def calculate_metrics(results, start_balance=START_BALANCE):
    """Calculate performance metrics."""
    years = (END_DATE - START_DATE).days / 365.25
    end_balance = results["Balance"].iloc[-1]
    ret = round(((end_balance / start_balance) - 1) * 100, 2)
    cagr = round(((end_balance / start_balance) ** (1 / years) - 1) * 100, 2)
    peak = results["Balance"].cummax()
    max_drawdown_pct = round(((results["Balance"] - peak) / peak).min() * 100, 2)
    rod = round(cagr / abs(max_drawdown_pct), 2)
    time_in_market = round(results["In_Market"].mean() * 100, 2)
    rbe = round((cagr / time_in_market) * 100, 2)
    rbeod = round(rbe / abs(max_drawdown_pct), 2)
    print(f"Start Balance: {start_balance:,}")
    print(f"Final Balance: {round(end_balance):,}")
    print(f"Total Return: {ret}%")
    print(f"Annual Return: {cagr}%")
    print(f"Max Drawdown: {max_drawdown_pct}%")
    print(f"Return over Drawdown: {rod}")
    print(f"Time in the Market: {time_in_market}%")
    print(f"Return By Exposure: {rbe}%")
    print(f"RBE over Drawdown: {rbeod}")
def export_detailed_data(dfs, results):
    """Export detailed trading data to CSV."""
    detailed_data = []
    for df in dfs:
        trade_data = df[['Open', 'High', 'Low', 'Close']].copy()
        trade_data['Consecutive_Higher_Highs'] = df['Consecutive_Higher_Highs']
        trade_data['Consecutive_Lower_Lows'] = df['Consecutive_Lower_Lows']
        trade_data['Long_Signal'] = df['Long_Signal']
        trade_data['Short_Signal'] = df['Short_Signal']
        trade_data['Previous_High_Broken'] = df['Previous_High_Broken']
        trade_data['Previous_Low_Broken'] = df['Previous_Low_Broken']
        trade_data['Long_Trade_Triggered'] = df['Long_Trade']
        trade_data['Short_Trade_Triggered'] = df['Short_Trade']
        trade_data['Long_Return'] = df['Long_Return']
        trade_data['Short_Return'] = df['Short_Return']
        trade_data['Daily_Return'] = np.where(
            trade_data['Long_Trade_Triggered'],
            trade_data['Long_Return'] - 1,
            np.where(
                trade_data['Short_Trade_Triggered'],
                trade_data['Short_Return'] - 1,
                0
            )
        )
        trade_data['In_Trade'] = (trade_data['Long_Trade_Triggered'] |
                                  trade_data['Short_Trade_Triggered'])
        trade_data['Trade_Type'] = np.where(
            trade_data['Long_Trade_Triggered'], 'LONG',
            np.where(trade_data['Short_Trade_Triggered'], 'SHORT', 'NO_TRADE')
        )
        detailed_data.append(trade_data)
    combined_data = pd.concat(detailed_data, keys=SYMBOLS, names=['Symbol', 'Date'])
    combined_data = combined_data.reset_index()
    combined_data = combined_data.merge(
        results['Balance'].reset_index(),
        on='Date',
        how='left'
    )
    filename = f"EURUSD_{START_DATE.date()}_{END_DATE.date()}.csv"
    combined_data.to_csv(filename, index=False)
    print(f"Detailed data exported to {filename}")
    return combined_data
if __name__ == "__main__":
    dfs = process_symbols()
    results = simulate_trading(dfs)
    plot_balance(results)
    calculate_metrics(results)
    detailed_data = export_detailed_data(dfs, results)