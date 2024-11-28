import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="darkgrid")

st.title("Advanced Interactive Financial Portfolio Optimizer")

# Sidebar Inputs
st.sidebar.header('User Inputs')

# Select assets
assets = st.sidebar.multiselect(
    'Select Assets (Choose at least 2)',
    ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NFLX', 'BRK-B', 'JNJ', 'JPM'],
    default=['AAPL', 'MSFT', 'GOOGL']
)

# Select start and end dates
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))

# Check date validity
if start_date >= end_date:
    st.sidebar.error('Error: End date must fall after start date.')

# Optimization objective
objective = st.sidebar.selectbox(
    'Optimization Objective',
    ['Maximize Sharpe Ratio', 'Minimize Volatility']
)

# Risk-free rate
risk_free_rate = st.sidebar.number_input('Risk-Free Rate (Annual %)', value=2.0) / 100

# Constraints
st.sidebar.header('Constraints')
allow_short = st.sidebar.checkbox('Allow Short Selling', value=False)
weight_bounds = (-1.0 if allow_short else 0.0, 1.0)

# Fetch data when assets are selected
if len(assets) >= 2:
    # Fetch adjusted closing prices
    data = yf.download(assets, start=start_date, end=end_date)['Adj Close']

    if data.isnull().values.any():
        st.warning('Some data might be missing. Please check the selected assets and date range.')

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252     # Annualized

    # Portfolio optimization functions
    def portfolio_performance(weights, mean_returns, cov_matrix):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
        return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

    def minimize_volatility(weights, mean_returns, cov_matrix):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    # Constraints and bounds
    num_assets = len(assets)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple([weight_bounds for _ in range(num_assets)])

    # Initial guess
    initial_guess = num_assets * [1. / num_assets,]

    # Optimization
    if objective == 'Maximize Sharpe Ratio':
        result = minimize(negative_sharpe_ratio, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(minimize_volatility, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x

    # Display results
    st.subheader('Optimized Portfolio Allocation')
    allocation = pd.DataFrame({'Asset': assets, 'Weight': optimal_weights})
    st.dataframe(allocation.set_index('Asset').style.format("{:.2%}"))

    # Portfolio performance
    p_ret, p_vol, p_sr = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    st.write(f"**Expected Annual Return:** {p_ret:.2%}")
    st.write(f"**Annual Volatility (Risk):** {p_vol:.2%}")
    st.write(f"**Sharpe Ratio:** {p_sr:.2f}")

    # Efficient frontier
    st.subheader('Efficient Frontier')

    def efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_portfolios=10000):
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.uniform(weight_bounds[0], weight_bounds[1], num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio
        return results, weights_record

    # Generate efficient frontier
    ef_results, ef_weights = efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(ef_results[0, :], ef_results[1, :], c=ef_results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    ax.scatter(p_vol, p_ret, marker='*', color='r', s=500, label='Optimal Portfolio')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility (Std. Deviation)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    fig.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    st.pyplot(fig)

else:
    st.warning('Please select at least two assets to proceed.')
