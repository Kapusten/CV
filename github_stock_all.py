import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def get_sp500_nasdaq100_symbols():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    nasdaq100_tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
    sp500_symbols = sp500['Symbol'].tolist()
    nasdaq100_symbols = []
    for table in nasdaq100_tables:
        if 'Symbol' in table.columns:
            nasdaq100_symbols = table['Symbol'].tolist()
            break
        elif 'Ticker' in table.columns:
            nasdaq100_symbols = table['Ticker'].tolist()
            break
    if not nasdaq100_symbols:
        print("Warning: Could not find Nasdaq-100 symbols. Check the Wikipedia page structure.")
    all_symbols = list(set(sp500_symbols + nasdaq100_symbols))
    return all_symbols
START_DATE = dt.datetime(2019, 1, 1)
END_DATE = dt.datetime(2024, 11, 6)
symbols = get_sp500_nasdaq100_symbols()
START_BALANCE = 10000
CONSECUTIVE_DAYS = 1
def get_price_data(symbol, start=START_DATE, end=END_DATE):
    symbol = symbol.replace('BF.B', 'BF-B').replace('BRK.B', 'BRK-B')
    df = yf.download(symbol, start, end,  progress=False).drop(columns=["Volume", "Adj Close"])
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
def process_symbols(symbols=symbols):
    """Fetch data, calculate indicators, and generate signals for each symbol."""
    return [generate_signals(calculate_indicators(get_price_data(symbol))) for symbol in symbols]
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
def simulate_trading_single(df, start_balance=START_BALANCE, start_date=START_DATE, end_date=END_DATE):
    """Simulate trading for a single symbol."""
    balances = [start_balance]
    dates = [start_date]
    in_market = [False]
    for day in df.index:
        if start_date <= day <= end_date:
            data = df.loc[day]
            if data["Long_Trade"] or data["Short_Trade"]:
                combined_ret = data["Long_Ret"] * data["Short_Ret"]
                new_balance = balances[-1] * combined_ret
                in_trade = True
            else:
                new_balance = balances[-1]
                in_trade = False
            balances.append(new_balance)
            dates.append(day)
            in_market.append(in_trade)
    return pd.DataFrame({
        "Date": dates,
        "Balance": balances,
        "In_Market": in_market
    }).set_index("Date")
def calculate_metrics_single(results, symbol, start_balance=START_BALANCE):
    """Calculate performance metrics for a single symbol."""
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
    return {
        "Symbol": symbol,
        "Final Balance": round(end_balance),
        "Total Return %": ret,
        "Max Drawdown %": max_drawdown_pct,
        "Time in Market %": time_in_market,
    }
def combine_and_analyze_all(dfs, symbols):
    """Analyze each symbol separately and combine results."""
    individual_results = []
    metrics_list = []
    for df, symbol in zip(dfs, symbols):
        results = simulate_trading_single(df)
        individual_results.append(results)
        metrics = calculate_metrics_single(results, symbol)
        metrics_list.append(metrics)
    combined_balance = pd.DataFrame()
    for i, result in enumerate(individual_results):
        if combined_balance.empty:
            combined_balance = result["Balance"].to_frame()
        else:
            combined_balance = combined_balance.join(result["Balance"], rsuffix=f'_{symbols[i]}')
    portfolio_balance = combined_balance.sum(axis=1)
    years = (END_DATE - START_DATE).days / 365.25
    total_start_balance = START_BALANCE * len(symbols)
    end_balance = portfolio_balance.iloc[-1]
    ret = round(((end_balance / total_start_balance) - 1) * 100, 2)
    cagr = round(((end_balance / total_start_balance) ** (1 / years) - 1) * 100, 2)
    peak = portfolio_balance.cummax()
    drawdown = (portfolio_balance - peak) / peak
    max_drawdown_pct = round(drawdown.min() * 100, 2)
    metrics_df = pd.DataFrame(metrics_list)
    sum_individual_drawdowns = metrics_df['Max Drawdown %'].sum()
    sum_individual_returns = metrics_df['Total Return %'].sum()
    portfolio_metrics = {
        "Symbol": "PORTFOLIO",
        "Final Balance": round(end_balance),
        "Total Return %": ret,
        "Sum of Individual Max Drawdowns %": round(sum_individual_drawdowns),
        "Sum of Individual Total Returns %": round(sum_individual_returns)
    }
    return metrics_df, portfolio_metrics, pd.DataFrame({"Balance": portfolio_balance})
dfs = process_symbols()
metrics_df, portfolio_metrics, portfolio_results = combine_and_analyze_all(dfs, symbols)
print("\nIndividual Symbol Performance:")
print(metrics_df)
print("\nPortfolio Performance:")
for key, value in portfolio_metrics.items():
    print(f"{key}: {value}")
