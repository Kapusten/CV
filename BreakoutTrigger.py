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
logger = logging.getLogger('breakout')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('breakout.txt')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
POPULAR_ETFS = [
    'QQQ', 'TQQQ', 'SPY', 'SPXL', 'DIA', 'IWM', 'EEM', 'VTI', 'GLD', 'SLV', 'GSY', 'IBIT',
    'XLF', 'VGT', 'VWO', 'IVV', 'TLT', 'XLK', 'VEA', 'VNQ', 'VOO',
    'SCHD', 'VYM', 'XLE', 'VIG', 'IEMG', 'IJR', 'VUG', 'XLV', 'VO', 'SMH',
    'VTV', 'FDN', 'BND', 'MUB', 'SPYV', 'BIL', 'IAU', 'QQQM',
    'SPLG', 'SCHX', 'XLY', 'XLI', 'XLC', 'IWB', 'VB', 'IWD', 'VBR', 'RSP',
    'SOXX', 'IGV', 'MTUM', 'COWZ', 'XLU', 'XOP', 'IWF', 'VOE', 'XBI',
    'PFF', 'FDIS', 'VBK', 'SPYG', 'XHB', 'VT', 'VHT', 'ITA', 'VTEB', 'SHY',
    'LQD', 'XME', 'EMB', 'IYR', 'VCSH', 'IGIB', 'GOVT', 'HYG', 'VCLT', 'TIP',
    'IEF', 'IHI', 'XRT', 'XLRE', 'EWJ', 'FXI', 'XPH', 'MGK', 'SCHA', 'XAR',
    'ESGU', 'XSD', 'IPAY', 'SKYY', 'SHE', 'FREL', 'FNDX', 'VGT', 'BIV', 'VNQI',
    'USO', 'DBC', 'UNG', 'PDBC', 'BNO', 'CORN', 'WEAT', 'SOYB', 'DBA', 'JO',
    'AGG', 'ITOT', 'USMV', 'QUAL', 'IJH', 'EFA', 'IEFA', 'SDY', 'DVY', 'XLP',
    'EEMV', 'DGRO', 'NOBL', 'VXUS', 'VDE', 'VSS', 'VPU', 'VDC', 'VFH', 'JETS',
    'DGRW', 'IGLB', 'VV', 'SCHF', 'SCHB', 'JPST', 'IGSB', 'ACWI', 'ICLN', 'BNDX',
    'VONG', 'EFAV', 'SPDW', 'SPEM', 'BSV', 'EWZ', 'MOAT', 'SCHG', 'SPHD', 'SPAB', '^NDX', '^GSPC', '^VIX',
    '^IXIC', '^RUT', 'CL=F', 'ZW=F', 'NG=F', 'GC=F',
]
data_cache = {}
def fetch_data(symbol, start_date, max_retries=30, delay=60):
    symbol = symbol.replace('BF.B', 'BF-B').replace('BRK.B', 'BRK-B')
    if (symbol, start_date) in data_cache:
        return data_cache[(symbol, start_date)]
    attempt = 0
    while attempt < max_retries:
        try:
            max_data = yf.download(symbol, period="max", progress=False)
            if not max_data.empty:
                max_available_date = max_data.index[0].strftime("%Y-%m-%d")
                if start_date < max_available_date:
                    start_date = max_available_date
                data = yf.download(symbol, start=start_date, progress=False)
                if not data.empty:
                    data_cache[(symbol, start_date)] = data
                    return data
                else:
                    raise ValueError("No data fetched, returned dataframe is empty.")
            else:
                raise ValueError("No data fetched for maximum available period.")
        except (ConnectionError, ReadTimeout) as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logger.error(f"All {max_retries} attempts failed. Could not download data for {symbol}.")
    return None
def write_to_csv(data, filename, mode='a'):
    with open(filename, mode, newline='') as csvfile:
        fieldnames = ['Symbol', 'Entry Date', 'Entry Price', 'Current Price', 'Current Trade Return', 'Strategy Return', 'Max Drawdown', 'Buy and Hold Return', 'Avg Profit Per Trade', 'Hold Period', 'Breakout Period', 'Max Hold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for row in data:
            writer.writerow(row)
def calculate_buy_and_hold(data):
    buy_price = data['Close'].iloc[0]
    sell_price = data['Close'].iloc[-1]
    buy_and_hold_return = (sell_price - buy_price) / buy_price * 100
    return buy_and_hold_return
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
    all_symbols = list(set( POPULAR_ETFS))
    return all_symbols
def identify_recent_exits(results, current_date):
    yesterday = current_date - pd.Timedelta(days=1)
    recent_exits = {'yesterday': [], 'today': []}
    for result in results:
        if result is not None and 'exit_dates' in result:
            for exit_date in result['exit_dates']:
                if exit_date.date() == current_date.date():
                    recent_exits['today'].append(result['Symbol'])
                elif exit_date.date() == yesterday.date():
                    recent_exits['yesterday'].append(result['Symbol'])
    return recent_exits
def breakout_strategy(data, breakout_periods, max_hold):
    data['Signal'] = 0
    data['ExitSignal'] = 0
    data['HoldPeriod'] = 0
    data['BreakoutPeriod'] = 0
    in_trade = False
    hold_count = 0
    for period in breakout_periods:
        data[f'High_{period}'] = data['High'].rolling(window=period).max().shift(1)
    for i in range(max(breakout_periods), len(data)):
        if not in_trade:
            breakout_occurred = False
            for period in breakout_periods:
                if data['High'].iloc[i] > data[f'High_{period}'].iloc[i]:
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
            if data['Close'].iloc[i] > data['High'].iloc[i - 1] or hold_count >= max_hold:
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
    entry_date = None
    exit_dates = []
    trade_profits = []
    breakout_period = 0
    for i in range(1, total_days):
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = data['Close'].iloc[i]
            entry_date = data.index[i]
            breakout_period = data['BreakoutPeriod'].iloc[i]
        elif data['ExitSignal'].iloc[i] == 1 and position == 1:
            position = 0
            exit_price = data['Close'].iloc[i]
            trade_return = (exit_price - entry_price) / entry_price
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
            exit_dates.append(data.index[i])
            trade_profits.append(trade_return * 100)
        if position == 1:
            invested_days += 1
        if len(equity_curve) > 1:
            max_equity = max(max_equity, equity_curve[-1])
    open_trade = None
    if position == 1:
        last_price = data['Close'].iloc[-1]
        unrealized_return = (last_price - entry_price) / entry_price
        current_drawdown = (max_equity - equity_curve[-1] * (1 + unrealized_return)) / max_equity * 100
        hold_period = (data.index[-1] - entry_date).days
        open_trade = {
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Current Date': data.index[-1],
            'Current Price': last_price,
            'Unrealized Return': unrealized_return * 100,
            'Total Profit': (equity_curve[-1] * (1 + unrealized_return) - 1) * 100,
            'Drawdown': current_drawdown,
            'Hold Period': hold_period,
            'Breakout Period': breakout_period
        }
    max_drawdown = ((np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)).max() * 100
    avg_profit_per_trade = np.mean(trade_profits) if trade_profits else 0
    return open_trade, max_drawdown, invested_days, total_days, equity_curve[-1], exit_dates, avg_profit_per_trade, len(trade_profits)
def breakout_search(start_date, breakout_periods, max_hold_periods):
    symbols = get_sp500_nasdaq100_symbols()
    logger.info(f"Parameters: Breakout Periods: {breakout_periods}, Max Hold Periods: {max_hold_periods}")
    args = [(symbol, start_date, breakout_period, max_hold)
            for symbol in symbols
            for breakout_period in breakout_periods
            for max_hold in max_hold_periods]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_stock, args)
    current_date = pd.Timestamp.now()
    results_by_period = {(bp, mh): [] for bp in breakout_periods for mh in max_hold_periods}
    for result in results:
        if result is not None:
            period = result['Breakout_Period']
            max_hold = result['Max_Hold']
            results_by_period[(period, max_hold)].append(result)
    all_specific_trades = []
    for (period, max_hold) in results_by_period.keys():
        period_results = results_by_period[(period, max_hold)]
        recent_exits = identify_recent_exits(period_results, current_date)
        open_trades = [trade for trade in period_results if
                       'Strategy Return' in trade and trade['Strategy Return'] > 10 and trade['Max Drawdown'] < 50]
        open_trades.sort(key=lambda x: (-x['Strategy Return']))
        logger.info(f"\n--- Results for Breakout Period {period} and Max Hold {max_hold} ---")
        specific_trades = [
            {
                'Symbol': trade['Symbol'],
                'Entry Date': trade['Entry Date'],
                'Entry Price': trade['Entry Price'],
                'Current Price': trade['Current Price'],
                'Current Trade Return': trade['Current Trade Return'],
                'Strategy Return': trade['Strategy Return'],
                'Max Drawdown': trade['Max Drawdown'],
                'Buy and Hold Return': trade['Buy and Hold Return'],
                'Avg Profit Per Trade': trade['Avg Profit Per Trade'],
                'Hold Period': trade['Hold Period'],
                'Breakout Period': period,
                'Max Hold': max_hold
            }
            for trade in open_trades
            if trade['Hold Period'] < 9 and trade['Strategy Return'] >= 40 and -3 < trade['Current Trade Return'] < 0.3 and trade['Max Drawdown'] < 35
        ]
        all_specific_trades.extend(specific_trades)
        logger.info("\nRecent Exits:")
        logger.info(f"Exited yesterday ({current_date.date() - pd.Timedelta(days=1)}): {', '.join(recent_exits['yesterday'])}")
        logger.info(f"Exited today ({current_date.date()}): {', '.join(recent_exits['today'])}")
    if all_specific_trades:
        filename = f"BreakoutTrigger_Strategy_{current_date.strftime('%Y%m%d')}.csv"
        write_to_csv(all_specific_trades, filename, mode='w')
        logger.info(f"\nSaved {len(all_specific_trades)} best trades to {filename}")
    end_time = time.time()
    logger.info(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
def process_stock(args):
    symbol, start_date, breakout_period, max_hold = args
    try:
        data = fetch_data(symbol, start_date)
        if data is None or len(data) < breakout_period:
            return None
        data = breakout_strategy(data, [breakout_period], max_hold)
        buy_and_hold_return = calculate_buy_and_hold(data)
        open_trade, max_drawdown, invested_days, total_days, final_equity, exit_dates, avg_profit_per_trade, num_trades = backtest_strategy(
            data, max_hold)
        result = {
            'Symbol': symbol,
            'Breakout_Period': breakout_period,
            'Max_Hold': max_hold,
            'exit_dates': exit_dates,
            'Avg Profit Per Trade': avg_profit_per_trade,
            'Number of Trades': num_trades
        }
        if open_trade:
            result.update(open_trade)
            result['Strategy Return'] = (final_equity - 1) * 100
            result['Buy and Hold Return'] = buy_and_hold_return
            result['Max Drawdown'] = max_drawdown
            result['Time Invested'] = (invested_days / total_days) * 100
            result['Hold Period'] = open_trade['Hold Period']
            result['Current Trade Return'] = open_trade['Unrealized Return']
            result['Breakout Period'] = open_trade['Breakout Period']
        return result
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return None
if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_date = '2015-01-01'
    breakout_periods = [5, 10, 15]
    max_hold_periods = [25, 20]
    breakout_search(start_date, breakout_periods, max_hold_periods)