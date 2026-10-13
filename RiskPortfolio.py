import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

risk_free_rate = 0.035



tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']

start_date='2020-01-01'
end_date ='2025-12-30'

adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if not data.empty and "Close" in data.columns:
        adj_close_df[ticker] = data["Close"]
    else:
        print(f"Warnung: Keine Daten für {ticker}")



log_returns=np.log(adj_close_df/adj_close_df.shift(1))

log_returns=log_returns.dropna()

cov_matrix = log_returns.cov()*252



def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_return (weights, log_returns)-risk_free_rate) / standard_deviation (weights, cov_matrix)

def neg_sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate)

constraints= {'type': 'eq', 'fun':lambda weights: np.sum(weights)-1}
bounds = [(0, 0.5) for _ in range (len(tickers))]

initial_weight = np.array([1/len(tickers)]*len(tickers))

optimized_results = minimize(neg_sharpe_ratio, initial_weight,args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
optimized_weights = optimized_results.x


print("Optimal Weights:")
for ticker, weight in zip(tickers, optimized_weights):
    print(f"{ticker}: {weight:.4f}")

print()

optimal_port_return = expected_return(optimized_weights, log_returns)
optimal_port_vola = standard_deviation(optimized_weights, cov_matrix)
optimal_port_sharpe = sharpe_ratio(optimized_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual return: {optimal_port_return: .4f}")
print(f"Expected volatility: {optimal_port_vola: .4f}")
print(f"Sharpe: {optimal_port_sharpe: .4f}")

plt.figure(figsize=(10,6))
plt.bar(tickers, optimized_weights)

plt.show()












































