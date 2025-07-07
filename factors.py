from sklearn.covariance import LedoitWolf
import numpy as np
import pandas as pd
from data import data as dt

class FactorModel:
  def __init__(self,R,X):
    self.R = R.values if isinstance(R, pd.DataFrame) else R  # (N x T)
    self.X = X.values if isinstance(X, pd.DataFrame) else X  # (N x K)


    self.f = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.R

  def compute_factor_covariance(self):
    T = self.f.shape[1]
    return (self.f@ self.f.T) / (T - 1)

  def compute_idiosyncratic_variance(self):
    eps = self.R - np.dot(self.X ,self.f)  # shape (N x T)
    variances = np.var(eps, axis=1, ddof=1)
    return np.diag(variances)

  def compute_total_risk_model(self, Sigma_f, D):
    return (self.X@Sigma_f @self.X.T)+ D

  def shrink_covariance(self):
      lw = LedoitWolf().fit(self.f.T)
      return lw.covariance_  # shape (K x K)

def factor():
    R=dt.stock_data.pct_change().dropna().T
    X=pd.DataFrame(columns=['beta','size','momentum','pe_ratio','volatility'])
    for ticker in dt.tickers:
        try:
            # Check if 'beta' data exists and is not empty for the ticker
            if ticker in dt.all_data and isinstance(dt.all_data[ticker], pd.DataFrame) and not dt.all_data[ticker]['beta'].empty:
                row = {
                    'beta': dt.all_data[ticker]['beta'].iloc[-1],
                    'size': dt.all_data[ticker]['size'].iloc[-1],
                    'momentum': dt.all_data[ticker]['momentum'].iloc[-1],
                    'pe_ratio': dt.all_data[ticker]['pe_ratio'].iloc[-1],
                    'volatility': dt.all_data[ticker]['volatility'].iloc[-1]
                }
            else:
                # Assign default values if 'beta' data is empty or missing
                row = {
                    'beta': 0,
                    'size': 0,
                    'momentum': 0,
                    'pe_ratio': 0,
                    'volatility': 0
                }
            X.loc[ticker] = row  # Add a row with index = ticker
        except KeyError:
            print(f"KeyError for ticker: {ticker}")
            # Assign default values in case of other KeyErrors
            row = {
                'beta': 0,
                'size': 0,
                'momentum': 0,
                'pe_ratio': 0,
                'volatility': 0
            }
            X.loc[ticker] = row
            continue


    X=X.fillna(0).astype(float)
    X = X.loc[R.index]

    factor=FactorModel(R,X)
    f=factor.f
    shrink=False
    Sigma_f = factor.shrink_covariance() if shrink else factor.compute_factor_covariance()
    D = factor.compute_idiosyncratic_variance()
    Sigma = factor.compute_total_risk_model(Sigma_f, D)