import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pymc3 as pm

# Define the stock ticker symbol
ticker = "BTCUSD"

# Fetch stock data from Yahoo Finance
stock_data = yf.Ticker(ticker).history(period="max")

# Create the combined chart
fig_combined = go.Figure()

# Add the candlestick chart
fig_combined.add_trace(go.Candlestick(x=stock_data.index,
                                      open=stock_data['Open'],
                                      high=stock_data['High'],
                                      low=stock_data['Low'],
                                      close=stock_data['Close'],
                                      name='Candlestick'))

# Add the volume chart
fig_combined.add_trace(go.Bar(x=stock_data.index,
                              y=stock_data['Volume'],
                              name='Volume',
                              marker_color='rgba(0, 0, 255, 0.3)'))

# Perform Bayesian regression on the price and volume data
price = stock_data['Close']
volume = stock_data['Volume']

with pm.Model() as model:
    # Define priors for regression coefficients
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    # Define likelihood
    mu = alpha + beta * volume
    sigma = pm.HalfCauchy('sigma', beta=10)
    price_observed = pm.Normal('price_observed', mu=mu, sd=sigma, observed=price)

    # Perform inference
    trace = pm.sample(2000, tune=1000, cores=1)
    posterior = pm.sample_posterior_predictive(trace)

# Plot the Bayesian regression line
x_range = np.linspace(volume.min(), volume.max(), 100)
y_mean = trace['alpha'].mean() + trace['beta'].mean() * x_range
y_hpd = pm.hpd(posterior['price_observed'], credible_interval=0.95)

fig_combined.add_trace(go.Scatter(x=x_range, y=y_mean, name='Bayesian Regression'))
fig_combined.add_trace(go.Scatter(x=np.concatenate([x_range, x_range[::-1]]),
                                  y=np.concatenate([y_hpd[:, 0], y_hpd[:, 1][::-1]]),
                                  fill='toself', fillcolor='rgba(0, 176, 246, 0.2)',
                                  line_color='rgba(0, 176, 246, 0.4)',
                                  name='95% Credible Interval'))

# Configure the layout
fig_combined.update_layout(
    title='Price and Volume Over Time with Bayesian Regression',
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True
)

# Show the combined chart
fig_combined.show()
