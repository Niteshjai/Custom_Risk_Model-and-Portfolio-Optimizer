import cvxpy as cp
import numpy as np
import pandas as pd
from factors import factor as ft
from data import data as dt

class Optimization:
  def __init__(self,tickers,sigma,num_sectors):
    self.tickers = tickers
    self.n = len(tickers)
    self.sigma = sigma


    # Define optimization variable
    self.w = cp.Variable((self.n, 1))

    # Define parameters (values will be set before solve)
    self.alpha = cp.Parameter((self.n, 1))
    self.beta = cp.Parameter((self.n, 1))
    self.w_prev = cp.Parameter((self.n, 1))
    self.sector_matrix = cp.Parameter((self.n,num_sectors))


  def objective_function(self):
    risk_aversion = 10
    return cp.Maximize(self.w.T @ self.alpha - risk_aversion * cp.quad_form(self.w, self.sigma))

  def constraints(self):
    turnover_limit = 10
    constraints = [
    cp.sum(self.w) == 1,                            # budget
    self.w.T@ self.beta == 0,                      # beta neutrality
    #self.w.T @ self.sector_matrix == 0,            # sector neutrality
    self.w >= 0,                                    # non-negativity
    cp.norm1(self.w - self.w_prev) <= turnover_limit  # turnover constraint
    ]
    return  constraints

  def solve_optimization(self, alpha, beta, sector_matrix, w_prev):
      # Assign parameter values
      self.alpha.value = alpha
      self.beta.value = beta
      self.sector_matrix.value = sector_matrix
      self.w_prev.value = w_prev

      prob = cp.Problem(self.objective_function(), self.constraints())
      prob.solve()
      print("Problem status:", prob.status)
      return self.w.value

def opt():
    alpha = ft.R.mean(axis=1).values.reshape(-1, 1)                  # Shape (n,1)
    beta = ft.beta_dt.iloc[-1].fillna(0.1).values.reshape(-1, 1)       # Shape (n,1)
    sector_matrix = dt.sector_matrix   
    previous_weights = np.ones((len(dt.tickers), 1))/len(dt.tickers)      # Can be last period's weights
    Sigma = ft.Sigma                                                  # Your risk model covariance matrix

    opt_model = Optimization(dt.tickers, Sigma,dt.num_sectors)
    weights = opt_model.solve_optimization(
        alpha,
        beta,
        sector_matrix,      # Should be shape (n, 1)
        previous_weights    # Should be shape (n, 1)
    )