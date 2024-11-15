import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
from requests.exceptions import ConnectionError, ReadTimeout
import multiprocessing
import csv
start_time = time.time()
logger = logging.getLogger('mfi_strategy')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('breakout_backtesting_low.txt')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
POPULAR_ETFS = ['EURUSD=X']
data_cache = {}
def fetch_data(symbol, start_date, end_date, max_retries=30, delay=60):
    symbol = symbol.replace('BF.B', 'BF-B').replace('BRK.B', 'BRK-B')
    if (symbol, start_date, end_date) in data_cache:
        return data_cache[(symbol, start_date, end_date)]
    attempt = 0
    while attempt < max_retries:
        try:
            ticker = yf.Ticker(symbol)
            max_data = ticker.history(period="max")
            if max_data.empty:
                raise ValueError("No data available for this symbol.")
            max_data.index = max_data.index.tz_localize(None)
            actual_start = max(pd.to_datetime(start_date), max_data.index[0])
            actual_end = min(pd.to_datetime(end_date), max_data.index[-1])
            data = max_data.loc[actual_start:actual_end]
            if len(data) < 50:
                return None
            data_cache[(symbol, start_date, end_date)] = data
            return data
        except Exception as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed for {symbol}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logger.error(f"All {max_retries} attempts failed. Could not download data for {symbol}.")
    return None
def get_sp500_nasdaq100_symbols():
    all_symbols = list(set(POPULAR_ETFS))
    return all_symbols
def breakout_strategy(data, breakout_periods, max_hold):
    data['Signal'] = 0
    data['ExitSignal'] = 0
    data['HoldPeriod'] = 0
    data['BreakoutPeriod'] = 0
    in_trade = False
    hold_count = 0
    for period in breakout_periods:
        data[f'Low_{period}'] = data['Low'].rolling(window=period).min().shift(1)
    for i in range(max(breakout_periods), len(data)):
        if not in_trade:
            breakout_occurred = False
            for period in breakout_periods:
                if data['Close'].iloc[i] < data[f'Low_{period}'].iloc[i]:
                    breakout_occurred = True
                    data.loc[data.index[i], 'BreakoutPeriod'] = period
                    break
            if breakout_occurred:
                data.loc[data.index[i], 'Signal'] = 1
                in_trade = True
                hold_count = 1
                data.loc[data.index[i], 'HoldPeriod'] = hold_count
        elif in_trade:
            hold_count += 1
            data.loc[data.index[i], 'HoldPeriod'] = hold_count
            if data['Close'].iloc[i] < data['Low'].iloc[i - 1] or hold_count >= max_hold:
                data.loc[data.index[i], 'ExitSignal'] = 1
                in_trade = False
                hold_count = 0
    return data
def backtest_strategy(data, max_hold):
    position = 0
    entry_price = 0
    equity_curve = [1.0]
    max_equity = 1.0
    invested_days = 0
    total_days = len(data)
    trade_profits = []
    drawdowns = []
    for i in range(1, total_days):
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = data['Close'].iloc[i]
        elif data['ExitSignal'].iloc[i] == 1 and position == 1:
            position = 0
            exit_price = data['Close'].iloc[i]
            trade_return = (entry_price - exit_price) / entry_price
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
            trade_profits.append(trade_return * 100)
        if position == 1:
            invested_days += 1
        if len(equity_curve) > 1:
            max_equity = max(max_equity, equity_curve[-1])
            current_drawdown = (max_equity - equity_curve[-1]) / max_equity * 100
            drawdowns.append(current_drawdown)
    return {
        'Total Return': (equity_curve[-1] - 1) * 100,
        'Max Drawdown': max(drawdowns) if drawdowns else 0,
        'Avg Profit Per Trade': np.mean(trade_profits) if trade_profits else 0,
        'Number of Trades': len(trade_profits),
        'Win Rate': sum(1 for p in trade_profits if p > 0) / len(trade_profits) if trade_profits else 0,
        'Time Invested': (invested_days / total_days) * 100
    }
def process_stock(args):
    symbol, start_date, end_date, breakout_period, max_hold = args
    try:
        data = fetch_data(symbol, start_date, end_date)
        if data is None or len(data) < breakout_period:
            return None
        data = breakout_strategy(data, [breakout_period], max_hold)
        backtest_results = backtest_strategy(data, max_hold)
        result = {
            'Symbol': symbol,
            'Breakout_Period': breakout_period,
            'Max_Hold': max_hold
        }
        result.update(backtest_results)
        return result
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return None
def breakout_low(end_date, breakout_periods, max_hold_periods):
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    symbols = get_sp500_nasdaq100_symbols()
    logger.info(f"Parameters: End Date: {end_date}, Start Date: {start_date}, "
                f"Breakout Periods: {breakout_periods}, Max Hold Periods: {max_hold_periods}")
    args = [(symbol, start_date, end_date, breakout_period, max_hold)
            for symbol in symbols
            for breakout_period in breakout_periods
            for max_hold in max_hold_periods]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_stock, args)
    results_by_period = {}
    for result in results:
        if result is not None:
            key = (result['Breakout_Period'], result['Max_Hold'])
            if key not in results_by_period:
                results_by_period[key] = []
            results_by_period[key].append(result)
    summary_data = []
    for (period, max_hold), period_results in results_by_period.items():
        summary_row = {
            'Strategy': 'LL Strategy',
            'Breakout Period': period,
            'Max Hold': max_hold,
            'Total Return': np.sum([r['Total Return'] for r in period_results]),
            'Total Max Drawdown': np.sum([r['Max Drawdown'] for r in period_results]),
            'Average Return': np.mean([r['Total Return'] for r in period_results]),
            'Average Max Drawdown': np.mean([r['Max Drawdown'] for r in period_results]),
            'Average Win Rate': np.mean([r['Win Rate'] for r in period_results]),
            'Average Trades per Symbol': np.mean([r['Number of Trades'] for r in period_results])
        }
        summary_data.append(summary_row)
        logger.info(f"\n--- LL Summary for Breakout Period {period} and Max Hold {max_hold} ---")
        logger.info(f"Total Return: {summary_row['Total Return']:.2f}%")
        logger.info(f"Total Max Drawdown: {summary_row['Total Max Drawdown']:.2f}%")
        logger.info(f"Average Return: {summary_row['Average Return']:.2f}%")
        logger.info(f"Average Max Drawdown: {summary_row['Average Max Drawdown']:.2f}%")
        logger.info(f"Average Win Rate: {summary_row['Average Win Rate']:.2f}%")
        logger.info(f"Average Trades per Symbol: {summary_row['Average Trades per Symbol']:.1f}")
    with open('strategy_summary_low.csv', 'w', newline='') as csvfile:
        fieldnames = ['Strategy', 'Breakout Period', 'Max Hold', 'Total Return', 'Total Max Drawdown',
                      'Average Return', 'Average Max Drawdown', 'Average Win Rate', 'Average Trades per Symbol']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
def backtest_strategy2(data, max_hold):
    position = 0
    entry_price = 0
    equity_curve = [1.0]
    max_equity = 1.0
    invested_days = 0
    total_days = len(data)
    trade_profits = []
    drawdowns = []
    for i in range(1, total_days):
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = data['Close'].iloc[i]
        elif data['ExitSignal'].iloc[i] == 1 and position == 1:
            position = 0
            exit_price = data['Close'].iloc[i]
            trade_return = (entry_price - exit_price) / entry_price
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
            trade_profits.append(trade_return * 100)
        if position == 1:
            invested_days += 1
        if len(equity_curve) > 1:
            max_equity = max(max_equity, equity_curve[-1])
            current_drawdown = (max_equity - equity_curve[-1]) / max_equity * 100
            drawdowns.append(current_drawdown)
    return {
        'Total Return': (equity_curve[-1] - 1) * 100,
        'Max Drawdown': max(drawdowns) if drawdowns else 0,
        'Avg Profit Per Trade': np.mean(trade_profits) if trade_profits else 0,
        'Number of Trades': len(trade_profits),
        'Win Rate': sum(1 for p in trade_profits if p > 0) / len(trade_profits) if trade_profits else 0,
        'Time Invested': (invested_days / total_days) * 100
    }
def breakout_strategy2(data, breakout_periods, max_hold):
    data['Signal'] = 0
    data['ExitSignal'] = 0
    data['HoldPeriod'] = 0
    data['BreakoutPeriod'] = 0
    in_trade = False
    hold_count = 0
    for period in breakout_periods:
        data[f'Low_{period}'] = data['Low'].rolling(window=period).min().shift(1)
    for i in range(max(breakout_periods), len(data)):
        if not in_trade:
            breakout_occurred = False
            for period in breakout_periods:
                if data['Low'].iloc[i] < data[f'Low_{period}'].iloc[i]:
                    breakout_occurred = True
                    data.loc[data.index[i], 'BreakoutPeriod'] = period
                    break
            if breakout_occurred:
                data.loc[data.index[i], 'Signal'] = 1
                in_trade = True
                hold_count = 1
                data.loc[data.index[i], 'HoldPeriod'] = hold_count
        elif in_trade:
            hold_count += 1
            data.loc[data.index[i], 'HoldPeriod'] = hold_count
            if data['Close'].iloc[i] < data['Low'].iloc[i - 1] or hold_count >= max_hold:
                data.loc[data.index[i], 'ExitSignal'] = 1
                in_trade = False
                hold_count = 0
    return data
def process_stock2(args):
    symbol, start_date, end_date, breakout_period, max_hold = args
    try:
        data = fetch_data(symbol, start_date, end_date)
        if data is None or len(data) < breakout_period:
            return None
        data = breakout_strategy2(data, [breakout_period], max_hold)
        backtest_results = backtest_strategy2(data, max_hold)
        result = {
            'Symbol': symbol,
            'Breakout_Period': breakout_period,
            'Max_Hold': max_hold
        }
        result.update(backtest_results)
        return result
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return None
def breakout_searchWaluty(end_date, breakout_periods, max_hold_periods):
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    symbols = get_sp500_nasdaq100_symbols()
    logger.info(f"Parameters: End Date: {end_date}, Start Date: {start_date}, "
                f"Breakout Periods: {breakout_periods}, Max Hold Periods: {max_hold_periods}")
    args = [(symbol, start_date, end_date, breakout_period, max_hold)
            for symbol in symbols
            for breakout_period in breakout_periods
            for max_hold in max_hold_periods]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_stock2, args)
    results_by_period = {}
    for result in results:
        if result is not None:
            key = (result['Breakout_Period'], result['Max_Hold'])
            if key not in results_by_period:
                results_by_period[key] = []
            results_by_period[key].append(result)
    summary_data = []
    for (period, max_hold), period_results in results_by_period.items():
        summary_row = {
            'Strategy': 'LL TRIGGER Strategy',
            'Breakout Period': period,
            'Max Hold': max_hold,
            'Total Return': np.sum([r['Total Return'] for r in period_results]),
            'Total Max Drawdown': np.sum([r['Max Drawdown'] for r in period_results]),
            'Average Return': np.mean([r['Total Return'] for r in period_results]),
            'Average Max Drawdown': np.mean([r['Max Drawdown'] for r in period_results]),
            'Average Win Rate': np.mean([r['Win Rate'] for r in period_results]),
            'Average Trades per Symbol': np.mean([r['Number of Trades'] for r in period_results])
        }
        summary_data.append(summary_row)
        logger.info(f"\n--- LL TRIGGER Summary for Breakout Period {period} and Max Hold {max_hold} ---")
        logger.info(f"Total Return: {summary_row['Total Return']:.2f}%")
        logger.info(f"Total Max Drawdown: {summary_row['Total Max Drawdown']:.2f}%")
        logger.info(f"Average Return: {summary_row['Average Return']:.2f}%")
        logger.info(f"Average Max Drawdown: {summary_row['Average Max Drawdown']:.2f}%")
        logger.info(f"Average Win Rate: {summary_row['Average Win Rate']:.2f}%")
        logger.info(f"Average Trades per Symbol: {summary_row['Average Trades per Symbol']:.1f}")
    with open('strategy_summary_low_trigger.csv', 'w', newline='') as csvfile:
        fieldnames = ['Strategy', 'Breakout Period', 'Max Hold', 'Total Return', 'Total Max Drawdown',
                      'Average Return', 'Average Max Drawdown', 'Average Win Rate', 'Average Trades per Symbol']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
if __name__ == "__main__":
    multiprocessing.freeze_support()
    end_date = '2024-09-01'
    breakout_periods = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 250]
    max_hold_periods = [1, 2, 3, 5]
    breakout_low(end_date, breakout_periods, max_hold_periods)
    breakout_searchWaluty(end_date, breakout_periods, max_hold_periods)
