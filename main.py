import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from factors import factor as ft
from data import data as dt
from optimizer import opt

def frontier():
    mus = np.linspace(-0.02, 0.05, 50)
    rets, risks = [], []

    for mu in mus:
        w = cp.Variable((len(dt.tickers), 1))
        constraints = [cp.sum(w) == 1, w >= 0, w <= 0.1, w.T @ opt.alpha >= mu]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, ft.Sigma)), constraints)
        prob.solve()
        if w.value is not None:
            rets.append(float(w.value.T @ opt.alpha))
            risks.append(np.sqrt(float(w.value.T @ ft.Sigma @ w.value)))

    plt.figure(figsize=(8,5))
    plt.plot(risks, rets)
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.grid(True)
    plt.show()

def return_plot():
    plt.figure(figsize=(12,6))
    plt.plot(cumulative_returns, label="Optimized")
    plt.plot(cum_equal, label="Equal Weight")
    plt.plot(cum_mc, label="Market Cap Weight")
    plt.title("Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def sharpe_ratio(rets, rf=0):
    return (rets.mean() - rf) / rets.std()

def max_drawdown(cum_rets):
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    return drawdown.min()

if __name__=="__main__" :
    portfolio_return=ft.R.T@opt.weights
    cumulative_returns=(1+portfolio_return).cumprod()

    equal_weights = np.ones(len(dt.tickers)) / len(dt.tickers)
    # Extract the last market cap value for each ticker, handling empty Series
    market_caps = np.array([dt.all_data[t]['size'].iloc[-1] if not dt.all_data[t]['size'].empty else 0 for t in dt.tickers]).reshape(-1,1)
    mc_weights = market_caps / np.nansum(market_caps)

    equal_returns = ft.R.T @ equal_weights
    mc_returns = ft.R.T @ mc_weights

    cum_equal = (1 + equal_returns).cumprod()
    cum_mc = (1 + mc_returns).cumprod()

    print("Sharpe (Optimized):", sharpe_ratio(portfolio_return))
    print("Volatility (Optimized):", portfolio_return.std())
    print("Max Drawdown:", max_drawdown(cumulative_returns))
    
    return_plot()
    frontier()