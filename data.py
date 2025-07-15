import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import zscore
from collections import Counter

# --- Collects and computes normalized factor data for a list of tickers ---
def collect_data(tickers, market_data, stock_data, period):
    all_data = {}  # Dictionary to store factor data for each stock

    market_returns = market_data['^GSPC'].pct_change().dropna()  # Daily market returns

    # Initialize empty DataFrames to collect each factor across all tickers
    beta_dt = pd.DataFrame(columns=tickers)
    size_dt = pd.DataFrame(columns=tickers)
    momentum_dt = pd.DataFrame(columns=tickers)
    pe_ratio_dt = pd.DataFrame(columns=tickers)
    volatility_dt = pd.DataFrame(columns=tickers)

    for ticker in tickers:
        if ticker in stock_data.columns:
            stock_returns = stock_data[ticker].pct_change().dropna()

            # Align stock and market returns for beta computation
            returns = pd.DataFrame({
                'Stocks': stock_returns,
                'market': market_returns
            }).dropna()

            # --- Compute Beta using rolling covariance and variance ---
            try:
                rolling_window = 30
                beta = returns['Stocks'].rolling(rolling_window).cov(returns['market']) / \
                       returns['market'].rolling(rolling_window).var()
            except:
                beta = pd.Series(index=returns.index, data=np.nan)

            # --- Compute Size (log market capitalization = log(price × shares_outstanding)) ---
            try:
                info = yf.Ticker(ticker).info
                outstanding_share = info['sharesOutstanding']
                price = stock_data[ticker]
                if outstanding_share is not None and price is not None:
                    size = np.log(outstanding_share * price)
                else:
                    size = pd.Series(index=returns.index, data=np.nan)
            except:
                size = pd.Series(index=returns.index, data=np.nan)

            # --- Compute Momentum (12-month return 1 month ago) ---
            try:
                momentum = (stock_data[ticker].shift(21) / stock_data[ticker].shift(252)) - 1
            except:
                momentum = pd.Series(index=stock_data[ticker].index, data=np.nan)

            # --- Get P/E ratio (static value from Yahoo Finance) ---
            try:
                info = yf.Ticker(ticker).info
                pe_ratio = info.get('trailingPE', np.nan)
            except:
                pe_ratio = np.nan

            # --- Compute Volatility (21-day rolling standard deviation) ---
            try:
                volatility = returns['Stocks'].rolling(21).std()
            except:
                volatility = pd.Series(index=returns.index, data=np.nan)

            # Store all computed factors into a ticker-specific DataFrame
            all_data[ticker] = pd.DataFrame({
                'returns': returns['Stocks'],
                'beta': beta,
                'size': size,
                'momentum': momentum,
                'pe_ratio': pe_ratio,
                'volatility': volatility
            })

            all_data[ticker].dropna(inplace=True)  # Keep only rows where all values are valid

            # Collect unnormalized data for Z-scoring
            beta_dt[ticker] = all_data[ticker]['beta']
            size_dt[ticker] = all_data[ticker]['size']
            momentum_dt[ticker] = all_data[ticker]['momentum']
            pe_ratio_dt[ticker] = all_data[ticker]['pe_ratio']
            volatility_dt[ticker] = all_data[ticker]['volatility']

        else:
            # If no stock data, assign NaNs
            all_data[ticker] = {
                'returns': np.nan,
                'beta': np.nan,
                'size': np.nan,
                'momentum': np.nan,
                'pe_ratio': np.nan,
                'volatility': np.nan
            }

    # --- Z-score normalization of each factor across time (column-wise) ---
    beta_dt_normalized = beta_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
    size_dt_normalized = size_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
    momentum_dt_normalized = momentum_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
    pe_ratio_dt_normalized = pe_ratio_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
    volatility_dt_normalized = volatility_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)

    # --- Update ticker-specific DataFrames with normalized factors ---
    for ticker in tickers:
        if ticker in all_data and isinstance(all_data[ticker], pd.DataFrame):
            all_data[ticker]['beta'] = beta_dt_normalized[ticker].loc[all_data[ticker].index]
            all_data[ticker]['size'] = size_dt_normalized[ticker].loc[all_data[ticker].index]
            all_data[ticker]['momentum'] = momentum_dt_normalized[ticker].loc[all_data[ticker].index]
            all_data[ticker]['pe_ratio'] = pe_ratio_dt_normalized[ticker].loc[all_data[ticker].index]
            all_data[ticker]['volatility'] = volatility_dt_normalized[ticker].loc[all_data[ticker].index]

    return all_data, beta_dt_normalized


# --- Downloads data and builds sector exposure vector ---
def data():
    tickers = [ ... ]  # Replace with list of tickers (e.g., S&P 100 constituents)
    period = '5Y'
    market = '^GSPC'  # S&P 500 index as market proxy

    # --- Download market and stock data ---
    market_data = yf.download(market, period=period, interval='1d')['Close']
    stock_data = yf.download(tickers, period=period, interval='1d')['Close']

    # --- Compute normalized factor data ---
    all_data, beta_dt = collect_data(tickers, market_data, stock_data, period)

    sector_map = {}  # Dictionary to hold sector info for each ticker

    # --- Populate sector map using Yahoo Finance info ---
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector_map[ticker] = info.get('sector', 'Unknown')
        except Exception as e:
            sector_map[ticker] = 'Unknown'

    # --- Find the most common (majority) sector ---
    sector_counts = Counter(sector_map.values())
    main_sector = sector_counts.most_common(1)[0][0]

    # --- Build a binary sector exposure vector ---
    # 1 if stock is in majority sector, -1 otherwise
    sector_vector = np.array([
        1 if sector_map[t] == main_sector else -1
        for t in tickers
    ]).reshape(-1, 1)  # Shape (n_tickers, 1)

    return all_data, sector_vector, beta_dt,stock_data,tickers
