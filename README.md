# Factor-Based Portfolio Optimization

## Installation

To use this project, you'll need to have the following dependencies installed:

- Python 3.x
- pandas
- numpy
- matplotlib
- yfinance
- scipy
- collections
- sklearn
- cvxpy

You can install the required packages using pip:

```
pip install pandas numpy matplotlib yfinance scipy scikit-learn cvxpy
```

## Usage

The main functionality of this project is to collect and compute normalized factor data for a list of tickers, and then use a factor model to optimize a portfolio based on the computed factors.

The entry point of the project is the `main.py` file. Here's an example of how to use the code:

```python
from main import return_plot

# Run the optimization and plot the results
return_plot()
```

This will generate a plot comparing the performance of the optimized portfolio, an equal-weighted portfolio, and a market-cap-weighted portfolio.

## API

The project consists of the following main modules:

1. `data.py`: Responsible for collecting and computing the normalized factor data for a list of tickers.
2. `factors.py`: Implements the factor model, including the computation of factor returns, factor covariance, and idiosyncratic variance.
3. `optimizer.py`: Defines the optimization problem and solves for the optimal portfolio weights.
4. `main.py`: The entry point of the project, which calls the other modules and generates the performance plots.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open a new issue or submit a pull request. Contributions are welcome!

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

The project does not currently include any automated tests. However, you can run the `main.py` script to verify the functionality of the portfolio optimization process.
