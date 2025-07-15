import cvxpy as cp
import numpy as np
import pandas as pd
from factors import factor 
from data import data 

class Optimization:
    def __init__(self, tickers, sigma):
        self.tickers = tickers
        self.n = len(tickers)  # Number of assets
        self.sigma = sigma     # Covariance matrix for risk model (n x n)

        # Optimization variable: portfolio weights vector (n x 1)
        self.w = cp.Variable((self.n, 1))

        # Parameters to be set before solving optimization
        self.alpha = cp.Parameter((self.n, 1))      # Expected returns vector
        self.beta = cp.Parameter((self.n, 1))       # Market beta exposures vector
        self.w_prev = cp.Parameter((self.n, 1))     # Previous period's portfolio weights (for turnover constraint)
        self.sector_vec = cp.Parameter((self.n, 1)) # Sector exposure vector (optional for constraints/extensions)

    def objective_function(self):
        risk_aversion = 10  # Risk aversion coefficient; higher means more risk-averse

        # Objective: maximize expected return minus risk penalty (quadratic form)
        return cp.Maximize(self.w.T @ self.alpha - risk_aversion * cp.quad_form(self.w, self.sigma))

    def constraints(self):
        turnover_limit = 0.05  # Maximum allowed turnover between periods
        epsilon = 0.01         # (Unused currently; can be for small tolerances)

        constraints = [
            cp.sum(self.w) == 1,               # Fully invested portfolio (weights sum to 1)
            self.w.T @ self.beta == 0,         # Beta neutrality constraint (market-neutral)
            self.w >= 0,                       # Long-only positions (no short selling)
            cp.norm1(self.w - self.w_prev) <= turnover_limit  # Turnover constraint: L1 norm limits change in weights
        ]
        return constraints

    def solve_optimization(self, alpha, beta, sector_vec, w_prev):
        # Assign parameter values before solving
        self.alpha.value = alpha
        self.beta.value = beta
        self.sector_vec.value = sector_vec
        self.w_prev.value = w_prev

        # Define the optimization problem with objective and constraints
        prob = cp.Problem(self.objective_function(), self.constraints())

        # Solve the optimization problem
        prob.solve()

        # Print problem status for debugging
        print("Problem status:", prob.status)

        # Return optimized weights as numpy array
        return self.w.value

def opt():
    all_data, sector_vector, beta_dt,stock_data,tickers=data()
    R,beta_dt,sector_vector,Sigma=factor()
    # Expected returns vector (mean returns per asset)
    alpha = R.mean(axis=1).values.reshape(-1, 1)  
    
    # Latest beta values with missing values filled by 0.1
    beta = beta_dt.iloc[-1].fillna(0.1).values.reshape(-1, 1)
    
    # Sector vector from data module, reshaped as column vector
    sector_vec = sector_vector.reshape(-1, 1)  
    
    # Initial portfolio weights (equal weight)
    previous_weights = np.ones((len(tickers), 1)) / len(tickers)
    
    # Covariance matrix for risk model
    Sigma = Sigma  
    
    # Instantiate the optimization model
    opt_model = Optimization(tickers, Sigma)
    
    # Run optimization repeatedly (e.g. for backtesting or iterative updates)
    for _ in range(100):
        weights = opt_model.solve_optimization(
            alpha,
            beta,
            sector_vec,
            previous_weights
        )
        # Update previous_weights for next iteration
        previous_weights = weights

    return weights