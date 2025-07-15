import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from factors import factor 
from data import data 
from optimizer import opt

def return_plot():
    plt.figure(figsize=(12,6))
    plt.plot(cumulative_returns, label="Optimized")      # Plot optimized portfolio cumulative returns
    plt.plot(cum_equal, label="Equal Weight")            # Plot equal-weighted portfolio cumulative returns
    plt.plot(cum_mc, label="Market Cap Weight")          # Plot market-cap-weighted portfolio cumulative returns
    plt.title("Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def sharpe_ratio(portfolio_returns, rf=0.0, periods_per_year=252):
    """
    Compute annualized Sharpe ratio from daily returns.
    rf: risk-free rate (annualized)
    """
    excess_returns = portfolio_returns - rf / periods_per_year
    ann_excess_return = np.mean(excess_returns) * periods_per_year
    ann_volatility = np.std(portfolio_returns) * np.sqrt(periods_per_year)
    return ann_excess_return / ann_volatility

def max_drawdown(cum_rets):
    peak = cum_rets.cummax()                              # Running maximum of cumulative returns
    drawdown = (cum_rets - peak) / peak                   # Drawdown series
    return drawdown.min()                                  # Maximum drawdown (largest drop)

if __name__=="__main__":

    all_data, sector_vector, beta_dt,stock_data,tickers=data()
    R,beta_dt,sector_vector,Sigma=factor()
    weights=opt()

    # Compute portfolio daily returns using optimized weights from the factor model
    portfolio_return = R.T @ opt.weights
    cumulative_returns = (1 + portfolio_return).cumprod() # Cumulative product to get total return series

    # Equal-weighted portfolio weights
    equal_weights = np.ones(len(tickers)) / len(tickers)
    
    # Market cap weights using latest size factor (log market cap)
    market_caps = np.array([
        all_data[t]['size'].iloc[-1] if not all_data[t]['size'].empty else 0
        for t in tickers
    ]).reshape(-1, 1)
    mc_weights = market_caps / np.nansum(market_caps)

    # Calculate daily returns for equal weight and market cap portfolios
    equal_returns = R.T @ equal_weights
    mc_returns = R.T @ mc_weights

    # Compute cumulative returns
    cum_equal = (1 + equal_returns).cumprod()
    cum_mc = (1 + mc_returns).cumprod()

    # Print performance metrics for optimized portfolio
    print("Sharpe (Optimized):", sharpe_ratio(portfolio_return))
    print("Volatility (Optimized):", portfolio_return.std())
    print("Max Drawdown:", max_drawdown(cumulative_returns))
    
    # Plot cumulative return comparison
    return_plot()
