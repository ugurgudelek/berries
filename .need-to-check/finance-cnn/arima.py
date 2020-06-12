
# https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# todo: learn confidence interval output.

"""

SARIMAX -> Seasonal AutoRegressive Integrated Moving Averages with eXogenous regressors 

ARIMA(p, d, q):
---------------
    'p' is the auto-regressive part of the model. It allows us to incorporate the effect of past values into our model. 
Intuitively, this would be similar to stating that it is likely to be warm tomorrow if it has been warm the past 3 days. 
    'd' is the integrated part of the model. This includes terms in the model that incorporate the amount of differencing 
(i.e. the number of past time points to subtract from the current value) to apply to the time series. Intuitively, 
this would be similar to stating that it is likely to be same temperature tomorrow if the difference in temperature 
in the last three days has been very small. 
    'q' is the moving average part of the model. This allows us to set the error of our model as a linear combination of 
the error values observed at previous time points in the past. 

Dealing with seasonal effects -> seasonal ARIMA -> ARIMA(p,d,q)(P,D,Q)s:
------------------------------------------------------------------------ 
    (p, d, q) are the non-seasonal parameters described above 
    (P, D, Q) follow the same definition but are applied to the seasonal component of the time series 
    s is the periodicity of the time series (4 for quarterly periods, 12 for yearly periods, etc.). 

AIC (Akaike Information Criterion): 
-----------------------------------
    measures how well a model fits the data while taking into account the overall complexity of the model.
A model that fits the data very well while using lots of features will be assigned a larger AIC score than a model 
that uses fewer features to achieve the same goodness-of-fit. 

"""

data = sm.datasets.co2.load_pandas()
y = data.data

# The 'MS' string groups the data in buckets by start of the month
y = y['co2'].resample('MS').mean()

# The term bfill means that we use the value before filling in missing values
y = y.fillna(y.bfill())

print(y)

y.plot(figsize=(15, 6))
plt.show()

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# Grid Search
# -----------
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


warnings.filterwarnings("ignore") # specify to ignore warning messages
def param_search():

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue


# Fitting an ARIMA Time Series Model
# ----------------------------------
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

# The coef column shows the weight (i.e. importance) of each feature and how each one impacts the time series.
# The P>|z| column informs us of the significance of each feature weight.
# Here, each weight has a p-value lower or close to 0.05, so it is reasonable to retain all of them in our model.
print(results.summary().tables[1])

# When fitting seasonal ARIMA models (and any other models for that matter),
# it is important to run model diagnostics to ensure that none of the assumptions made by the model have been violated.
# The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.
results.plot_diagnostics(figsize=(15, 12))
plt.show()
# Our primary concern is to ensure that the residuals of our model are uncorrelated and
# normally distributed with zero-mean

#       In the top right plot, we see that the red KDE line follows closely with the N(0,1) line (where N(0,1)) is
# the standard notation for a normal distribution with mean 0 and standard deviation of 1). This is a good indication
# that the residuals are normally distributed.
#       The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) follows the linear
# trend of the samples taken from a standard normal distribution with N(0, 1). Again, this is a strong indication
# that the residuals are normally distributed.
#       The residuals over time (top left plot) don't display any obvious seasonality and appear to be white noise.
# This is confirmed by the autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time
# series residuals have low correlation with lagged versions of itself.

# Validating Forecasts
# --------------------
# (Dynamic -> False)
# Prediction and confidence interval
# The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point
# are generated using the full history up to that point.
pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()  # confidence interval

ax = y['1990':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

# Mean Squared Error
# ------------------
y_forecasted = pred.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# (Dynamic -> True)
# a better representation of our true predictive power can be obtained using dynamic forecasts.
# In this case, we only use information from the time series up to a certain point, and after that,
# forecasts are generated using values from previous forecasted time points
pred_dynamic = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y['1990':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1],
                 alpha=.1, zorder=-1)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

# Producing and Visualizing Forecasts
# -----------------------------------
# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()