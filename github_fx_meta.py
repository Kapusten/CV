import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
START_BALANCE = 10000
CONSECUTIVE_DAYS = 1
def load_csv_data(file_path):
    """Load and prepare historical data from CSV file."""
    df = pd.read_csv(file_path,
                     delimiter='\t',
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'],
                     skiprows=1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d')
    df = df.set_index('Date')
    df = df.drop(['TickVol', 'Vol', 'Spread'], axis=1)
    return df
def calculate_indicators(df):
    """Calculate daily indicators and consecutive highs/lows."""
    df["oc"] = df["Close"] / df["Open"]
    df["hc"] = df["Close"] / df["High"].shift(1)
    df["lc"] = df["Close"] / df["Low"].shift(1)
    df['LH'] = df["High"] < df["High"].shift(1)
    df['HL'] = df["Low"] > df["Low"].shift(1)
    df['Cons_Down'] = df['LH'] * (df['LH'].groupby((df['LH'] != df['LH'].shift()).cumsum()).cumcount() + 1)
    df['Cons_Up'] = df['HL'] * (df['HL'].groupby((df['HL'] != df['HL'].shift()).cumsum()).cumcount() + 1)
    return df
def generate_signals(df, consecutive_days=CONSECUTIVE_DAYS):
    """Generate trading signals and calculate returns for long and short trades."""
    df["Long_Signal"] = df['Cons_Down'] >= consecutive_days
    df["Short_Signal"] = df['Cons_Up'] >= consecutive_days
    df["High_Broken"] = (df["Open"] > df["High"].shift(1)) | (df["High"] >= df["High"].shift(1))
    df["Low_Broken"] = (df["Open"] < df["Low"].shift(1)) | (df["Low"] <= df["Low"].shift(1))
    df["Long_Trade"] = df["Long_Signal"].shift(1) & df["High_Broken"]
    df["Short_Trade"] = df["Short_Signal"].shift(1) & df["Low_Broken"]
    df['Long_Ret'] = np.where(df["Long_Trade"],
                              np.where(df["Open"] > df["High"].shift(1), df["oc"], df["hc"]),
                              1)
    df['Short_Ret'] = np.where(df["Short_Trade"] & ~df["Long_Trade"],
                               np.where(df["Open"] < df["Low"].shift(1), 2 - df["oc"], 2 - df["lc"]),
                               1)
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
def simulate_trading(df, start_balance=START_BALANCE):
    """Simulate trading with the generated signals."""
    balance = start_balance
    balances = []
    in_market = []
    for idx, row in df.iterrows():
        if row["Long_Trade"] or row["Short_Trade"]:
            returns = row["Long_Ret"] * row["Short_Ret"]
            balance *= returns
            in_market.append(True)
        else:
            in_market.append(False)
        balances.append(balance)
    results = pd.DataFrame({
        "Balance": balances,
        "In_Market": in_market
    }, index=df.index)
    return results
def calculate_metrics(results, start_balance=START_BALANCE):
    """Calculate performance metrics."""
    years = (results.index[-1] - results.index[0]).days / 365.25
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
if __name__ == "__main__":
    csv_file_path = "eurusd.csv"
    df = load_csv_data(csv_file_path)
    df = calculate_indicators(df)
    df = generate_signals(df)
    results = simulate_trading(df)
    plot_balance(results)
    calculate_metrics(results)