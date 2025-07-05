import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import zscore
from collections import Counter

def collect_data(tickers,market_data,stock_data, period):
  all_data = {}

  market_returns = market_data['^GSPC'].pct_change().dropna()
  beta_dt = pd.DataFrame(columns=tickers)
  size_dt= pd.DataFrame(columns=tickers)
  momentum_dt = pd.DataFrame(columns=tickers)
  pe_ratio_dt= pd.DataFrame(columns=tickers)
  volatility_dt= pd.DataFrame(columns=tickers)

  for ticker in tickers:
      # Handle potential KeyError if a ticker's data is missing
      if ticker in stock_data.columns:
        stock_returns = stock_data[ticker].pct_change().dropna()

        # Align returns Series by date
        returns= pd.DataFrame({
            'Stocks': stock_returns,
            'market': market_returns
        }).dropna()


        # beta=cov(stock,market)/var(market)
        try:
          rolling_window = 30  # Can change to 90, 126 etc.
          beta = returns['Stocks'].rolling(rolling_window).cov(returns['market']) / \
                returns['market'].rolling(rolling_window).var()
        except:
          beta = pd.Series(index=returns.index, data=np.nan)

        # Size=log(market cap)=log(outstanding share * price)
        try:
          info = yf.Ticker(ticker).info
          outstanding_share = info['sharesOutstanding']
          price = stock_data[ticker]
          if outstanding_share is not None and price is not None:
            size = np.log(outstanding_share * price)
          else:
            size = pd.Series(index=returns.index,data=np.nan)
        except:
          size = pd.Series(index=returns.index,data=np.nan)


        # --- Momentum (Price 1M ago to 12M ago) ---
        try:
            momentum = (stock_data[ticker].shift(21) / stock_data[ticker].shift(252)) - 1
        except:
            momentum = pd.Series(index=stock_data[ticker].index,data=np.nan)


        # P/E ratio
        try:
          info=yf.Ticker(ticker).info
          pe_ratio=info.get('trailingPE',np.nan)
        except:
          pe_ratio=np.nan


        # --- Volatility (21-day rolling std of returns) ---
        try:
            volatility = returns['Stocks'].rolling(21).std()
        except:
            volatility = pd.Series(index=returns.index,data=np.nan)

        all_data[ticker] =pd.DataFrame({
                    'returns': returns['Stocks'],
                    'beta': beta,
                    'size': size,
                    'momentum': momentum,
                    'pe_ratio': pe_ratio,
                    'volatility': volatility
        })

        all_data[ticker].dropna(inplace=True)
        # Ensure dates are aligned before assigning
        beta_dt[ticker]=all_data[ticker]['beta']
        size_dt[ticker]=all_data[ticker]['size']
        momentum_dt[ticker]=all_data[ticker]['momentum']
        pe_ratio_dt[ticker]=all_data[ticker]['pe_ratio']
        volatility_dt[ticker]=  all_data[ticker]['volatility']

      else:
         all_data[ticker] = {
              'returns': np.nan,
              'beta': np.nan,
              'size': np.nan,
              'momentum': np.nan,
              'pe_ratio': np.nan,
              'volatility': np.nan
          }
  # Apply z-score normalization across all tickers (i.e., column-wise)
  # Use .apply(lambda x: zscore(x, nan_policy='omit'), axis=1) to handle NaNs
  beta_dt_normalized = beta_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
  size_dt_normalized = size_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
  momentum_dt_normalized = momentum_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
  pe_ratio_dt_normalized = pe_ratio_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
  volatility_dt_normalized = volatility_dt.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)


  # Now update all_data[ticker] DataFrames with normalized values
  for ticker in tickers:
    if ticker in all_data and isinstance(all_data[ticker], pd.DataFrame):
          # Ensure the index is aligned before assigning normalized data
          all_data[ticker]['beta'] = beta_dt_normalized[ticker].loc[all_data[ticker].index]
          all_data[ticker]['size'] = size_dt_normalized[ticker].loc[all_data[ticker].index]
          all_data[ticker]['momentum'] = momentum_dt_normalized[ticker].loc[all_data[ticker].index]
          all_data[ticker]['pe_ratio'] = pe_ratio_dt_normalized[ticker].loc[all_data[ticker].index]
          all_data[ticker]['volatility'] = volatility_dt_normalized[ticker].loc[all_data[ticker].index]


  return all_data,beta_dt_normalized

def data():
  tickers = [
      "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "UNH",
      "JNJ", "V", "XOM", "JPM", "PG", "LLY", "MA", "HD", "CVX", "MRK",
      "PEP", "ABBV", "AVGO", "COST", "KO", "WMT", "MCD", "BAC", "ADBE", "CSCO",
      "PFE", "CRM", "ACN", "INTC", "TMO", "VZ", "ABT", "NFLX", "NKE", "DHR",
      "ORCL", "LIN", "TXN", "NEE", "AMGN", "UPS", "MS", "QCOM", "PM", "BMY",
      "IBM", "AMAT", "SBUX", "RTX", "CAT", "MDT", "HON", "GE", "GS", "LOW",
      "CVS", "INTU", "UNP", "PLD", "DE", "NOW", "SPGI", "ISRG", "MDLZ", "ADP",
      "LRCX", "BKNG", "SYK", "BLK", "CI", "T", "ZTS", "SCHW", "EL", "GILD",
      "MU", "ADI", "MO", "MMC", "FI", "PNC", "BDX", "ICE", "SO", "EW",
      "USB", "C", "APD", "CL", "ITW", "ETN", "FDX", "ADSK", "CSX", "AON"
  ]
  period='5Y'
  market='^GSPC'
  market_data = yf.download(market, period=period, interval='1d')['Close']
  stock_data = yf.download(tickers, period=period, interval='1d')['Close']

  all_data,beta_dt=collect_data(tickers,market_data,stock_data, period)

  # Dictionary to hold sector info
  sector_map = {}

  # Loop over tickers and fetch sector info
  for ticker in tickers:
      try:
          stock = yf.Ticker(ticker)
          info = stock.info
          sector_map[ticker] = info.get('sector', 'Unknown')
      except Exception as e:
          sector_map[ticker] = 'Unknown'

  sectors = sorted(set(sector_map.values()))
  num_sectors = len(sectors)
  sector_exposure_matrix = []

  for t in tickers:
      one_hot = [1 if sector_map[t] == s else 0 for s in sectors]
      sector_exposure_matrix.append(one_hot)

  # Convert to numpy matrix (shape: n_assets x n_sectors)
  sector_matrix = np.array(sector_exposure_matrix)  # shape (n, k)
