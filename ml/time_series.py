import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from common import data_wrangling as dw
from typing import List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Time series analysis of momentum - how the momentum of a certain market predicts future performance of that market
# Test for stationarity

#_OUT_PATH = Path("out/sarimax")
#_SEED = 42


def stationary_test(index: List[str])-> pd.DataFrame:
    data = dw.all_data()
    y = data[index]

    # Test stationarity
    result = adfuller(y)
    if result[1] > 0.05:
        # Data is non-stationary, apply differencing
        print('non-stationary!')
        y = y.diff().dropna()
    return y

def seasonal_decomposition(index: List[str]):
    data = dw.all_data()
    data.index = pd.to_datetime(data.index)
    y = pd.Series(data[index], index=data.index)
    y = y.asfreq('B')
    x = pd.Series(data[f'{index}_MOM'], index=data.index)
    x = x.asfreq('B')
    result = seasonal_decompose(y, model='additive', period=252)  # Assuming yearly seasonality

    # Plot the decomposed components
    result.plot()
    plt.show()


def sarimax_lags(index: List[str]):
    data = dw.all_data()
    data.index = pd.to_datetime(data.index)
    y = pd.Series(data[index], index=data.index)
    y = y.asfreq('B')
    x = pd.Series(data[f'{index}_MOM'], index=data.index)
    x = x.asfreq('B')


    # Initialize variables
    best_aic = float('inf')
    best_bic = float('inf')
    best_order = None

    # Loop over AR and MA lags
    for p in range(3):
        for q in range(3):
            # Fit SARIMAX model
            model = SARIMAX(y, exog=x, order=(p, 0, q), seasonal_order=(1, 1, 1, 252))
            result = model.fit()

            # Compare AIC and BIC
            if result.aic < best_aic and result.bic < best_bic:
                best_aic = result.aic
                best_bic = result.bic
                best_order = (p, 0, q)

    # Print the best model order
    print('Best model order:', best_order)
    """ Result:
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
    Best model order: (2, 0, 2)
    """
    return

def sarimax(index: List[str]):
    data = dw.all_data()
    data.index = pd.to_datetime(data.index)
    y_original = pd.Series(data[index], index=data.index)
    y = y.diff().dropna()
    y = y.asfreq('B')
    x = pd.Series(data[f'{index}_MOM'], index=data.index)
    x = x.asfreq('B')
    x = x.loc[y.index]

    data_length = len(y)
    train_size = int(data_length * 0.8)  # using 80% of the data for training

    y_train = y[:train_size]
    y_val = y[train_size:]

    # Exogenous variable, momentum
    x_train = x[:train_size]
    x_val = x[train_size:]

    model = SARIMAX(y_train, exog=x_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 252))
    result = model.fit()

    forecast = result.get_forecast(steps=len(y_val), exog=x_val)
    mean_forecast = forecast.predicted_mean

    #Reverse differencing
    last_training_value = y_original.iloc[train_size]

    # Initialize an empty list to store the reconstructed values
    reconstructed = [last_training_value]

    # Reverse differencing using the forecasted values
    for diff in mean_forecast:
        next_value = reconstructed[-1] + diff
        reconstructed.append(next_value)

    reconstructed_series = pd.Series(reconstructed[1:])  # [1:] to exclude the initial training value

    with open("sarimax_result.pkl", "wb") as file:
        pickle.dump(result, file)

    with open("sarimax_summary.txt", "w") as file:
        file.write(result.summary().as_text())

    metrics = {
        "AIC": result.aic,
        "BIC": result.bic,
        "HQIC": result.hqic,
        "p-values": result.pvalues,
        "COEFFS": result.params
    }

    pd.DataFrame([metrics]).to_csv("sarimax_metrics.csv", index=False)

    df_results = pd.DataFrame({
        "Actual": y_val,
        "Predicted": mean_forecast,
        "Residuals": y_val - mean_forecast
    })

    df_results.to_csv("sarimax_predictions.csv")

    mae = mean_absolute_error(y_val, mean_forecast)
    rmse = mean_squared_error(y_val, mean_forecast, squared=False)
    print(f"MAE: {mae}, RMSE: {rmse}")

    print(result.summary())
    #result.plot_diagnostics(figsize=(15, 12))

    p_values = result.pvalues
    print(p_values)

    coefficients = result.params
    print(coefficients)

    aic_value = result.aic
    print("AIC:", aic_value)

    # To get the BIC value:
    bic_value = result.bic
    print("BIC:", bic_value)



    # get p-values, if small, can reject null hypothesis, that is: no correlation between momentum and stock performance
    # Coefficient size and magnitude
    # MOdel fit metrics, AIC BIC, low values better, but beware of overfitting
    # Diagnostic plots, residuals, are there patterns left?
    # Out-of-sample forecasting, split data into training and test set
    # COmpare predicted values and actual values using:
    # Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE).
    # model validation, compare to actual outcomes.

def plot_residuals(residuals: pd.Series):
    # Access residuals
    #residuals = result.resid

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title('Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.show()


