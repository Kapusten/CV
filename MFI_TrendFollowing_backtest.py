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
file_handler = logging.FileHandler('mfi_trendFollowing_backtested.txt')
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
    'XLF', 'VGT', 'VWO', 'IVV', 'TLT', 'XLK', 'VEA', 'VNQ', 'ARKK', 'VOO',
    'SCHD', 'VYM', 'XLE', 'VIG', 'IEMG', 'IJR', 'VUG', 'XLV', 'VO', 'SMH',
    'VTV', 'FDN', 'BND', 'MUB', 'SPYV', 'ARKG', 'ARKW', 'BIL', 'IAU', 'QQQM',
    'SPLG', 'SCHX', 'XLY', 'XLI', 'XLC', 'IWB', 'VB', 'IWD', 'VBR', 'RSP',
    'SOXX', 'IGV', 'MTUM', 'COWZ', 'XLU', 'XOP', 'IWF', 'ARKF', 'VOE', 'XBI',
    'PFF', 'FDIS', 'VBK', 'SPYG', 'XHB', 'VT', 'VHT', 'ITA', 'VTEB', 'SHY',
    'LQD', 'XME', 'EMB', 'IYR', 'VCSH', 'IGIB', 'GOVT', 'HYG', 'VCLT', 'TIP',
    'IEF', 'IHI', 'XRT', 'XLRE', 'EWJ', 'FXI', 'XPH', 'MGK', 'SCHA', 'XAR',
    'ESGU', 'XSD', 'IPAY', 'SKYY', 'SHE', 'FREL', 'FNDX', 'VGT', 'BIV', 'VNQI',
    'USO', 'DBC', 'UNG', 'PDBC', 'BNO', 'CORN', 'WEAT', 'SOYB', 'DBA', 'JO',
    'AGG', 'ITOT', 'USMV', 'QUAL', 'IJH', 'EFA', 'IEFA', 'SDY', 'DVY', 'XLP',
    'EEMV', 'DGRO', 'NOBL', 'VXUS', 'VDE', 'VSS', 'VPU', 'VDC', 'VFH', 'JETS',
    'DGRW', 'IGLB', 'VV', 'SCHF', 'SCHB', 'JPST', 'IGSB', 'ACWI', 'ICLN', 'BNDX',
    'VONG', 'EFAV', 'SPDW', 'SPEM', 'BSV', 'EWZ', 'MOAT', 'SCHG', 'SPHD', 'SPAB', '^NDX', '^GSPC', '^VIX',
    '^IXIC', '^RUT', 'BTC-USD', 'CL=F', 'ZW=F', 'NG=F', 'GC=F',
 ]
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
def calculate_mfi(data, period):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi
def mfi_strategy(data, mfi_period, mfi_trigger, max_hold):
    data['MFI'] = calculate_mfi(data, mfi_period)
    data['Signal'] = 0
    data['ExitSignal'] = 0
    data['HoldPeriod'] = 0
    in_trade = False
    for i in range(1, len(data)):
        if data['MFI'].iloc[i] > mfi_trigger and not in_trade:
            data.loc[data.index[i], 'Signal'] = 1
            in_trade = True
            data.loc[data.index[i], 'HoldPeriod'] = 1
        elif in_trade:
            data.loc[data.index[i], 'HoldPeriod'] = data['HoldPeriod'].iloc[i - 1] + 1
        if data['Close'].iloc[i] > data['High'].iloc[i - 1] and in_trade:
            data.loc[data.index[i], 'ExitSignal'] = 1
            in_trade = False
            data.loc[data.index[i], 'HoldPeriod'] = 0
        if data['HoldPeriod'].iloc[i] >= max_hold:
            data.loc[data.index[i], 'ExitSignal'] = 1
            in_trade = False
            data.loc[data.index[i], 'HoldPeriod'] = 0
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
        elif (data['ExitSignal'].iloc[i] == 1 or data['HoldPeriod'].iloc[i] >= max_hold) and position == 1:
            position = 0
            exit_price = data['Close'].iloc[i]
            trade_return = (exit_price - entry_price) / entry_price
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
            trade_profits.append(trade_return * 100)
        if position == 1:
            invested_days += 1
        if len(equity_curve) > 1:
            max_equity = max(max_equity, equity_curve[-1])
            current_drawdown = (max_equity - equity_curve[-1]) / max_equity * 100
            drawdowns.append(current_drawdown)
    total_return = (equity_curve[-1] - 1) * 100
    max_drawdown = max(drawdowns) if drawdowns else 0
    avg_profit_per_trade = np.mean(trade_profits) if trade_profits else 0
    num_trades = len(trade_profits)
    win_rate = sum(1 for profit in trade_profits if profit > 0) / num_trades if num_trades > 0 else 0
    return {
        'Total Return': total_return,
        'Max Drawdown': max_drawdown,
        'Avg Profit Per Trade': avg_profit_per_trade,
        'Number of Trades': num_trades,
        'Win Rate': win_rate,
        'Time Invested': (invested_days / total_days) * 100
    }
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
    all_symbols = list(set(sp500_symbols + nasdaq100_symbols + POPULAR_ETFS))
    return all_symbols
def process_stock(args):
    symbol, start_date, end_date, mfi_period, mfi_trigger, max_hold = args
    try:
        data = fetch_data(symbol, start_date, end_date)
        if data is None or len(data) < 100:
            return None
        data = mfi_strategy(data, mfi_period, mfi_trigger, max_hold)
        backtest_results = backtest_strategy(data, max_hold)
        result = {
            'Symbol': symbol,
            'MFI_Period': mfi_period,
            'Max_Hold': max_hold,
            'MFI_Trigger': mfi_trigger
        }
        result.update(backtest_results)
        return result
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return None
def mfi_search(end_date, mfi_periods, mfi_trigger, max_hold_periods):
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=6)).strftime('%Y-%m-%d')
    symbols = get_sp500_nasdaq100_symbols()
    logger.info(
        f"Parameters: End Date: {end_date}, Start Date: {start_date}, MFI Periods: {mfi_periods}, MFI Trigger: {mfi_trigger}, Max Hold Periods: {max_hold_periods}")
    args = [(symbol, start_date, end_date, mfi_period, mfi_trigger, max_hold)
            for symbol in symbols
            for mfi_period in mfi_periods
            for max_hold in max_hold_periods]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_stock, args)
        all_results = [result for result in results if result is not None]
        results_by_period_and_hold = {}
        for result in all_results:
            key = (result['MFI_Period'], result['Max_Hold'])
            if key not in results_by_period_and_hold:
                results_by_period_and_hold[key] = []
            results_by_period_and_hold[key].append(result)
        for (mfi_period, max_hold), period_hold_results in results_by_period_and_hold.items():
            logger.info(f"\n--- Summary for MFI Period {mfi_period} and Max Hold {max_hold} ---")
            total_return = np.sum([r['Total Return'] for r in period_hold_results])
            total_max_drawdown = np.sum([r['Max Drawdown'] for r in period_hold_results])
            avg_return = np.mean([r['Total Return'] for r in period_hold_results])
            avg_drawdown = np.mean([r['Max Drawdown'] for r in period_hold_results])
            max_max_drawdown = np.max([r['Max Drawdown'] for r in period_hold_results])
            avg_profit_per_trade = np.mean([r['Avg Profit Per Trade'] for r in period_hold_results])
            total_trades = np.sum([r['Number of Trades'] for r in period_hold_results])
            avg_win_rate = np.mean([r['Win Rate'] for r in period_hold_results])
            logger.info(f"Total Return: {total_return:.2f}%")
            logger.info(f"Total Max Drawdown: {total_max_drawdown:.2f}%")
            logger.info(f"Average Total Return: {avg_return:.2f}%")
            logger.info(f"Average Max Drawdown: {avg_drawdown:.2f}%")
            logger.info(f"Max Max Drawdown: {max_max_drawdown:.2f}%")
            logger.info(f"Average Profit Per Trade: {avg_profit_per_trade:.2f}%")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Average Win Rate: {avg_win_rate:.2f}%")
        end_time = time.time()
        logger.info(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    end_date = '2010-12-31'
    mfi_periods = [2, 3, 25]
    max_hold_periods = [10, 15, 20]
    mfi_search(end_date, mfi_periods, 80, max_hold_periods)
