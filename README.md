# 📈 Custom Risk Model & Portfolio Optimizer

A Python-based toolkit to build customized risk models and optimize portfolios using historical financial data, modern portfolio theory, and advanced risk metrics.

---

## 🧠 Core Components

### `data.py`

* Fetches 5 years of daily close prices for selected stocks and the S&P 500 index using Yahoo Finance (yfinance).
* For each stock, calculates the following factors:

  * `Beta (30-day rolling)`
  * `Size (log of market cap)`
  * `Momentum (12-month return 1 month ago)`
  * `P/E Ratio`
  * `Volatility (21-day rolling std dev)`
    
* Applies Z-score normalization to each factor across time for all tickers.
* Determines the most common sector among tickers.
* Creates a vector: +1 for stocks in that sector, −1 otherwise, for use in portfolio constraints or analysis.



### `factors.py`

* Builds a factor model using asset returns and custom factor exposures (`beta`, `size`, `momentum`, `pe_ratio`, `volatility`).
* Estimates:

  * Factor returns via OLS
  * Factor covariance matrix
  * Idiosyncratic (specific) variance
  * Total covariance matrix using:

    $$
    \Sigma = X \Sigma_f X^\top + D
    $$

### `optimizer.py`

* Defines an `Optimization` class using `cvxpy`:

  * **Objective**: maximize return minus risk penalty
  * **Constraints**:

    * Fully invested (`sum(weights) = 1`)
    * Beta-neutral (`portfolio beta = 0`)
    * Long-only
    * Turnover constraint (`||w_t - w_{t-1}||_1 ≤ 0.05`)
* Uses expected returns, beta exposure, and sector exposure as parameters.
* Solves the optimization iteratively to simulate rebalancing.

### `main.py`

* Loads historical stock and factor data via the `data()` function.
* Computes portfolio returns for:

  * Optimized weights
  * Equal-weighted portfolio
  * Market cap-weighted portfolio
* Evaluates performance via:

  * Sharpe Ratio
  * Volatility
  * Maximum Drawdown
* Plots cumulative returns of all three strategies.

---

---

## 🧰 Tech Stack

* **Python**: core logic, data handling
* **Pandas**, **NumPy**, **SciPy**: data processing & statistical computation
* **Matplotlib** / **Seaborn** / **Plotly**: charts & visualizations
* **CVXOPT** or **SciPy Optimize**: for constrained optimizations
* **Jupyter Notebook**: interactive analysis environment

---

## 📁 Project Structure

```
Custom_Risk_Model-and-Portfolio-Optimizer/
├── factor.py              	# Builds covariance, risk metrics, factor models
├── optimizer.py               # Contains optimization routines (Sharpe, var-min)
├── main.py                	# Flow of program,calculate sharpe,max_drawdown and plot returns
├── data/
│   └── *.csv                  # Historical price & returns data
├── requirements.txt           # Required Python packages
└── README.md                  # This documentation file
```

---


## 🧪 Usage

### 1. Requirements

* `cvxpy`
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`

### 2. Run the portfolio optimizer

```bash
python main.py
```

This will:

* Build the factor model and compute the risk matrix.
* Solve the optimization problem.
* Evaluate performance and display cumulative return plot.

---


---

## ⚙️ Customization

* Modify risk metrics in `risk_model.py` (e.g., include CVaR, drawdown risk).
* Add objective types in `optimizer.py` (`min_variance`, `max_return`, etc.).
* Tune simulation parameters in `backtest.py` (Monte Carlo draws, time horizons).
* Adapt `visualize.py` to create customized plots (interactive via Plotly or Matplotlib).

---




---

## 📌 Notes

* Factor exposures must be available for each ticker.
* Missing values are handled with defaults (e.g., zeros or 0.1 for betas).

---

## 👨‍💼 Author

**Nitesh Jaiswal**

