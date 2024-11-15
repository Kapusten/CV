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
file_handler = logging.FileHandler('mfi_trendFollowing.txt')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
def write_to_csv(data, filename, mode='a'):
    with open(filename, mode, newline='') as csvfile:
        fieldnames = ['Symbol', 'Entry Date', 'Entry Price', 'Current Price', 'Current Trade Return', 'Strategy Return',
                      'Max Drawdown', 'Buy and Hold Return', 'Avg Profit Per Trade', 'Hold Period', 'MFI Period',
                      'Max Hold', 'Current Regime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for row in data:
            writer.writerow(row)
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
    '^IXIC', '^RUT', 'BTC-USD', 'CL=F', 'ZW=F', 'NG=F', 'GC=F', 'ALE.WA', 'ALR.WA', 'BDX.WA', 'CDR.WA', 'CPS.WA', 'DNP.WA',
    'JSW.WA', 'KGH.WA', 'KRU.WA', 'KTY.WA', 'LPP.WA', 'MBK.WA', 'OPL.WA', 'PCO.WA', 'PEO.WA', 'PGE.WA', 'PKN.WA',
    'PKO.WA', 'PZU.WA', 'SPL.WA', '11B.WA', 'ABE.WA', 'ACP.WA', 'APR.WA', 'ASB.WA', 'ATC.WA', 'ATT.WA', 'BFT.WA', 'BHW.WA', 'BNP.WA', 'CAR.WA', 'CBF.WA', 'CCC.WA', 'CIG.WA', 'CMR.WA', 'DOM.WA', 'DVL.WA', 'EAT.WA', 'ENA.WA', 'EUR.WA', 'GEA.WA', 'GPP.WA', 'GPW.WA', 'GRX.WA', 'HUG.WA', 'ING.WA', 'LWB.WA', 'MBR.WA', 'MIL.WA', 'MRB.WA', 'NEU.WA', 'RBW.WA', 'RVU.WA', 'SLV.WA', 'SNT.WA', 'TEN.WA', 'TPE.WA', 'TXT.WA', 'WPL.WA', 'XTB.WA'
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
        except (ConnectionError, ReadTimeout, ValueError) as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
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
def calculate_volatility(data, window=5):
    returns = data['Close'].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility
def determine_regime(volatility, low_threshold=0.15, high_threshold=0.35):
    if volatility <= low_threshold:
        return 'low'
    elif volatility >= high_threshold:
        return 'high'
    else:
        return 'medium'
def get_mfi_threshold(regime):
    thresholds = {
        'low': 85,
        'medium': 80,
        'high': 75
    }
    return thresholds[regime]
def regime_switching_mfi_strategy(data, mfi_period, max_hold):
    data['Volatility'] = calculate_volatility(data)
    data['Regime'] = data['Volatility'].apply(lambda x: determine_regime(x) if pd.notnull(x) else 'medium')
    data['MFI'] = calculate_mfi(data, mfi_period)
    data['MFI_Threshold'] = data['Regime'].apply(get_mfi_threshold)
    data['Signal'] = 0
    data['ExitSignal'] = 0
    data['HoldPeriod'] = 0
    in_trade = False
    for i in range(1, len(data)):
        if data['MFI'].iloc[i] > data['MFI_Threshold'].iloc[i] and not in_trade:
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
    entry_date = None
    exit_dates = []
    hold_period = 0
    trade_profits = []
    for i in range(1, total_days):
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = data['Close'].iloc[i]
            entry_date = data.index[i]
            hold_period = 0
        elif (data['ExitSignal'].iloc[i] == 1 or hold_period >= max_hold) and position == 1:
            position = 0
            exit_price = data['Close'].iloc[i]
            trade_return = (exit_price - entry_price) / entry_price
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
            exit_dates.append(data.index[i])
            trade_profits.append(trade_return * 100)
            hold_period = 0
        if position == 1:
            invested_days += 1
            hold_period += 1
        if len(equity_curve) > 1:
            max_equity = max(max_equity, equity_curve[-1])
    open_trade = None
    if position == 1:
        last_price = data['Close'].iloc[-1]
        unrealized_return = (last_price - entry_price) / entry_price
        current_drawdown = (max_equity - equity_curve[-1] * (1 + unrealized_return)) / max_equity * 100
        hold_period = (data.index[-1] - entry_date).days
        current_regime = data['Regime'].iloc[-1]
        open_trade = {
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Current Date': data.index[-1],
            'Current Price': last_price,
            'Unrealized Return': unrealized_return * 100,
            'Total Profit': (equity_curve[-1] * (1 + unrealized_return) - 1) * 100,
            'Drawdown': current_drawdown,
            'Hold Period': hold_period,
            'Current Regime': current_regime
        }
    max_drawdown = ((np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(
        equity_curve)).max() * 100
    avg_profit_per_trade = np.mean(trade_profits) if trade_profits else 0
    return open_trade, max_drawdown, invested_days, total_days, equity_curve[-1], exit_dates, avg_profit_per_trade, len(
        trade_profits)
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
    all_symbols = list(set(sp500_symbols + nasdaq100_symbols + POPULAR_ETFS))
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
def process_stock(args):
    symbol, start_date, mfi_period, max_hold = args
    try:
        data = fetch_data(symbol, start_date)
        if data is None or len(data) < 100:
            return None
        data = regime_switching_mfi_strategy(data, mfi_period, max_hold)
        buy_and_hold_return = calculate_buy_and_hold(data)
        open_trade, max_drawdown, invested_days, total_days, final_equity, exit_dates, avg_profit_per_trade, num_trades = backtest_strategy(
            data, max_hold)
        result = {
            'Symbol': symbol,
            'MFI_Period': mfi_period,
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
            result['Current Regime'] = open_trade['Current Regime']
        return result
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return None
def mfi_search(start_date, mfi_periods, max_hold_periods):
    symbols = get_sp500_nasdaq100_symbols()
    logger.info(f"Parameters: MFI Periods: {mfi_periods}, Max Hold Periods: {max_hold_periods}")
    args = [(symbol, start_date, mfi_period, max_hold)
            for symbol in symbols
            for mfi_period in mfi_periods
            for max_hold in max_hold_periods]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_stock, args)
    current_date = pd.Timestamp.now()
    results_by_params = {(period, max_hold): [] for period in mfi_periods for max_hold in max_hold_periods}
    for result in results:
        if result is not None:
            period = result['MFI_Period']
            max_hold = result['Max_Hold']
            results_by_params[(period, max_hold)].append(result)
    all_specific_trades = []
    for period in mfi_periods:
        for max_hold in max_hold_periods:
            param_results = results_by_params[(period, max_hold)]
            recent_exits = identify_recent_exits(param_results, current_date)
            open_trades = [trade for trade in param_results if
                           'Strategy Return' in trade and trade['Strategy Return'] > 50 and trade['Max Drawdown'] < 30]
            open_trades.sort(key=lambda x: (-x['Strategy Return']))
            logger.info(f"\n--- Results for MFI Period {period} and Max Hold {max_hold} ---")
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
                    'MFI Period': period,
                    'Max Hold': max_hold,
                    'Current Regime': trade['Current Regime']
                }
                for trade in open_trades
                if 1 <= trade['Hold Period'] < 7 and trade['Strategy Return'] >= 70 and trade[
                    'Current Trade Return'] < -0.35 and trade['Max Drawdown'] < 30
            ]
            all_specific_trades.extend(specific_trades)
            logger.info("\nOpen Trades Summary:")
            for trade in open_trades:
                logger.info(f"\nSymbol: {trade['Symbol']}")
                logger.info(f"Entry Date: {trade['Entry Date']}")
                logger.info(f"Entry Price: ${trade['Entry Price']:.2f}")
                logger.info(f"Current Price: ${trade['Current Price']:.2f}")
                logger.info(f"Unrealized Return: {trade['Current Trade Return']:.2f}%")
                logger.info(f"Strategy Total Return: {trade['Strategy Return']:.2f}%")
                logger.info(f"Buy and Hold Return: {trade['Buy and Hold Return']:.2f}%")
                logger.info(f"Max Drawdown: {trade['Max Drawdown']:.2f}%")
                logger.info(f"Avg Profit Per Trade: {trade['Avg Profit Per Trade']:.2f}%")
                logger.info(f"Hold Period: {trade['Hold Period']} days")
                logger.info(f"Current Regime: {trade['Current Regime']}")
            logger.info(f"\nTotal open trades: {len(open_trades)}")
            logger.info("\nRecent Exits:")
            logger.info(
                f"Exited yesterday ({current_date.date() - pd.Timedelta(days=1)}): {', '.join(recent_exits['yesterday'])}")
            logger.info(f"Exited today ({current_date.date()}): {', '.join(recent_exits['today'])}")
    if all_specific_trades:
        filename = f"MFI_trendFollowing_regimes_{current_date.strftime('%Y%m%d')}.csv"
        write_to_csv(all_specific_trades, filename, mode='w')
        logger.info(f"\nSaved {len(all_specific_trades)} best trades to {filename}")
    end_time = time.time()
    logger.info(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_date = '2018-01-01'
    mfi_periods = [2, 3]
    max_hold_periods = [5, 10, 15, 20]
    mfi_search(start_date, mfi_periods, max_hold_periods)
