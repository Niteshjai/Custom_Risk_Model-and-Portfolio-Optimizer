from sklearn.covariance import LedoitWolf
import numpy as np
import pandas as pd
from data import data 

class FactorModel:
    def __init__(self, R, X):
        # R: Returns matrix (N assets x T time points), converted to numpy array if DataFrame
        # X: Factor exposures matrix (N assets x K factors), converted similarly
        self.R = R.values if isinstance(R, pd.DataFrame) else R  # shape (N x T)
        self.X = X.values if isinstance(X, pd.DataFrame) else X  # shape (N x K)

        # Estimate factor returns f = (X'X)^-1 X'R by OLS regression of returns on factors
        self.f = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.R  # shape (K x T)

    def compute_factor_covariance(self):
        # Compute sample covariance of factor returns f (K x T)
        T = self.f.shape[1]
        return (self.f @ self.f.T) / (T - 1)  # shape (K x K)

    def compute_idiosyncratic_variance(self):
        # Residuals: returns minus factor model predictions
        eps = self.R - np.dot(self.X, self.f)  # shape (N x T)

        # Compute variance of residuals for each asset (idiosyncratic risk)
        variances = np.var(eps, axis=1, ddof=1)  # shape (N,)
        return np.diag(variances)  # Return diagonal matrix (N x N)

    def compute_total_risk_model(self, Sigma_f, D):
        # Compute total covariance matrix Σ = X Σ_f X' + D
        # Σ_f: factor covariance matrix (K x K)
        # D: diagonal matrix of idiosyncratic variances (N x N)
        return (self.X @ Sigma_f @ self.X.T) + D  # shape (N x N)

    def shrink_covariance(self):
        # Use Ledoit-Wolf shrinkage estimator to get a better-conditioned factor covariance matrix
        lw = LedoitWolf().fit(self.f.T)
        return lw.covariance_  # shrunk covariance matrix (K x K)

def factor():
    all_data, sector_vector, beta_dt,stock_data,tickers=data()
    R = stock_data.pct_change().dropna().T

    # Create DataFrame X to hold factor exposures for each ticker (N x K factors)
    X = pd.DataFrame(columns=['beta', 'size', 'momentum', 'pe_ratio', 'volatility'])

    for ticker in tickers:
        try:
            # Check if factor data for ticker exists and beta column is not empty
            if ticker in all_data and isinstance(all_data[ticker], pd.DataFrame) and not all_data[ticker]['beta'].empty:
                row = {
                    'beta': all_data[ticker]['beta'].iloc[-1],
                    'size': all_data[ticker]['size'].iloc[-1],
                    'momentum': all_data[ticker]['momentum'].iloc[-1],
                    'pe_ratio': all_data[ticker]['pe_ratio'].iloc[-1],
                    'volatility': all_data[ticker]['volatility'].iloc[-1]
                }
            else:
                # Default zeros if factor data missing
                row = {
                    'beta': 0,
                    'size': 0,
                    'momentum': 0,
                    'pe_ratio': 0,
                    'volatility': 0
                }
            # Insert the row with ticker as index
            X.loc[ticker] = row
        except KeyError:
            # In case of any key errors, log and insert zeros
            print(f"KeyError for ticker: {ticker}")
            row = {
                'beta': 0,
                'size': 0,
                'momentum': 0,
                'pe_ratio': 0,
                'volatility': 0
            }
            X.loc[ticker] = row
            continue

    # Fill any remaining NaNs with zero and ensure float type
    X = X.fillna(0).astype(float)

    # Align X rows to match the order of returns matrix R (index = tickers)
    X = X.loc[R.index]

    # Instantiate the FactorModel with returns R and factor exposures X
    factor = FactorModel(R, X)
    f = factor.f  # Factor returns (K x T)

    shrink = False  # Flag to enable Ledoit-Wolf shrinkage

    # Compute factor covariance matrix, optionally shrunk
    Sigma_f = factor.shrink_covariance() if shrink else factor.compute_factor_covariance()

    # Compute idiosyncratic variance matrix D
    D = factor.compute_idiosyncratic_variance()

    # Compute total asset covariance matrix Sigma
    Sigma = factor.compute_total_risk_model(Sigma_f, D)

    return R,beta_dt,sector_vector,Sigma

    