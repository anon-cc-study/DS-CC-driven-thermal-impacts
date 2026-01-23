import numpy as np
import scipy
import pandas as pd
import time
import os
import re
import glob
import pwlf
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

def metrics_test_data(y_test, model_y_pred):
    # Convert to numpy arrays
    y_test = np.array(y_test)
    model_y_pred = np.array(model_y_pred)

    # ---Overall Metrics---
    r2 = sklearn.metrics.r2_score(y_test, model_y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_test, model_y_pred)
    mape = np.mean(np.abs((y_test - model_y_pred) / y_test)) * 100
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, model_y_pred))
    nrmse = (rmse / np.mean(y_test)) * 100

    # ---Peak-related Metrics---
    max_y_test = np.max(y_test)
    max_model_pred = np.max(model_y_pred)
    peak_load_error = ((max_y_test - max_model_pred) / max_y_test) * 100

    max_y_test_idx = np.argmax(y_test)
    corresponding_pred = model_y_pred[max_y_test_idx]
    peak_time_error = ((max_y_test - corresponding_pred) / max_y_test) * 100

    # ---High-Value Metrics (Above 75th Percentile)---
    threshold_75 = np.percentile(y_test, 75)
    high_value_mask = y_test > threshold_75
    y_test_high = y_test[high_value_mask]
    y_pred_high = model_y_pred[high_value_mask]

    if len(y_test_high) > 0:
        mape_high = np.mean(np.abs((y_test_high - y_pred_high) / y_test_high)) * 100
        rmse_high = np.sqrt(sklearn.metrics.mean_squared_error(y_test_high, y_pred_high))
        nrmse_high = (rmse_high / np.mean(y_test_high)) * 100
    else:
        mape_high = np.nan
        nrmse_high = np.nan

    return (
        r2, mae, mape, rmse, nrmse, peak_load_error, peak_time_error,
        mape_high, nrmse_high
    )


def stratified_temperature_split(X, y, num_bins, test_size, random_state):
    """
    Perform stratified sampling based on temperature bins to ensure 
    that the temperature distribution in the training and test sets is consistent.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex")
    
    # Get temperature data
    temperature = X["Dry Bulb Temperature [°C]"]
    
    # Create temperature stratification labels (e.g., divide into 10 bins based on quantiles)
    X["temp_bin"] = pd.qcut(temperature, q=num_bins, labels=False, duplicates="drop")

    # Perform stratified sampling based on temperature
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=X["temp_bin"], random_state=random_state)

    # Remove the temporary column temp_bin
    X_train = X_train.drop(columns=["temp_bin"])
    X_test = X_test.drop(columns=["temp_bin"])

    return X_train, X_test, y_train, y_test



def add_features_X_columns_D(df_original):
    """
    Input: weather data frame that has columns "Dry Bulb Temperature [°C]" and "hour"
    Function:
    Add to dataframe with Temp and hour columns) input features for X_columns_D:
      - "last 24h avg Temp"
      - "last 12h avg Temp"
      - "Minus 1h Temp"
      - "Minus 3h Temp"
      - "Minus 6h Temp"
      - "Minus 12h Temp"
      - "Minus 24h Temp"
      - "sin hour"
      - "cos hour"
      - "temp times sin hour"
      - "temp times cos hour"
    Output: data frame with the newly added columns
    """
    df = df_original
    
    temp_col = "Dry Bulb Temperature [°C]"
    # Rolling averages - using expanding window if not enough history
    df["last 24h avg Temp"] = df[temp_col].rolling(window=24, min_periods=1).mean()
    df["last 12h avg Temp"] = df[temp_col].rolling(window=12, min_periods=1).mean()

    # Lag features with fallback to nearest available past value
    for lag_hours in [1, 3, 6, 12, 24]:
        col_name = f"Minus {lag_hours}h Temp"
        shifted = df[temp_col].shift(lag_hours)
        # Fill NaN values in early rows with the most recent available (closest lag)
        for i in range(lag_hours):
            shifted.iloc[i] = df[temp_col].iloc[max(i - 1, 0)]
        df[col_name] = shifted

    # Time-based cyclical features
    df["sin hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Interaction features
    df["temp times sin hour"] = df[temp_col] * df["sin hour"]
    df["temp times cos hour"] = df[temp_col] * df["cos hour"]
    
    return df

def train_mlp_model(input_df, X_columns,Y_column,first_L_n, second_L_n): #df is the input; check_date are used to check the peaks
    train_start_time = time.time()
    # Define independent and dependent parameters 
    X = input_df[X_columns]
    y = input_df[[Y_column]]
    # Split data to training and testing 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = stratified_temperature_split(X, y, num_bins = 10, test_size = 0.2, random_state=123)
    # normalize x
    xnorm = MinMaxScaler().fit(X_train)
    X_train_norm = xnorm.transform(X_train)
    X_test_norm = xnorm.transform(X_test)
    # normalize y
    ynorm = MinMaxScaler().fit(y_train)
    y_train_norm = ynorm.transform(y_train)
    
    # Create a Multi-Layer Perceptron (MLP) regression model

    #  Create MLP model with 1 or 2 hidden layers
    if second_L_n == '' or second_L_n is None:
        hidden_layers = (first_L_n,)
    else:
        hidden_layers = (first_L_n, second_L_n)

    MLP_model = MLP(
        activation='relu',
        hidden_layer_sizes=hidden_layers,
        learning_rate='constant',
        max_iter=100,
        solver='adam',
        random_state=42
    )
    
    # Train the MLP model using the normalized training data
    MLP_model.fit(X_train_norm,y_train_norm)
    
    # MLP_score = cross_val_score(MLP_model, X_train_norm, y_train_norm, cv=5, scoring='r2')     
    
    # Predict y_test values using the trained model
    MLP_y_test_pred_norm = MLP_model.predict(X_test_norm).reshape(-1,1) # predict y and reshape it into a 2D array (reshape(-1,1)) for compatibility with inverse_transform
    MLP_y_test_pred = ynorm.inverse_transform(MLP_y_test_pred_norm) # denormalize MLP_y_pred_norm back to the original scale
    
    # Predict y_train and denormalize
    MLP_y_train_pred_norm = MLP_model.predict(X_train_norm).reshape(-1, 1)
    MLP_y_train_pred = ynorm.inverse_transform(MLP_y_train_pred_norm)
 
    # Convert component to lists
    y_test_values = y_test.values.flatten().tolist()  # Flatten array and convert to list
    y_test_pred = MLP_y_test_pred.flatten().tolist()
    y_test = y_test_values
    
    # Convert training predictions to list
    y_train_values = y_train.values.flatten().tolist()
    y_train_pred = MLP_y_train_pred.flatten().tolist()
    y_train = y_train_values
    
    train_end_time = time.time(); training_runtime = (train_end_time - train_start_time) / 60;
        
    return MLP_model, xnorm, ynorm, y_test, y_test_pred, y_train, y_train_pred, training_runtime



def train_linear_model(input_df, X_columns, Y_column):  # Renamed function to reflect Linear Regression
    train_start_time = time.time()
    # Define independent and dependent parameters
    X = input_df[X_columns]
    y = input_df[[Y_column]]
    # Split data to training and testing
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = stratified_temperature_split(X, y, num_bins = 10, test_size = 0.2, random_state=123)
    # normalize x
    xnorm = MinMaxScaler().fit(X_train)
    X_train_norm = xnorm.transform(X_train)
    X_test_norm = xnorm.transform(X_test)
    # normalize y
    ynorm = MinMaxScaler().fit(y_train)
    y_train_norm = ynorm.transform(y_train)

    # Create a Linear Regression model (CHANGED)
    linear_model = LinearRegression()  # CHANGED
    # Train the Linear Regression model using the normalized training data
    linear_model.fit(X_train_norm, y_train_norm)  # SAME

    # Predict y_test values using the trained model
    linear_y_test_pred_norm = linear_model.predict(X_test_norm).reshape(-1, 1)  # SAME
    linear_y_test_pred = ynorm.inverse_transform(linear_y_test_pred_norm)  # SAME

    # Convert component to lists
    y_test_values = y_test.values.flatten().tolist()  # SAME
    y_test_pred = linear_y_test_pred.flatten().tolist()  # SAME
    y_test = y_test_values

    train_end_time = time.time()
    training_runtime = (train_end_time - train_start_time) / 60

    return linear_model, xnorm, ynorm, y_test, y_test_pred, training_runtime


def train_piecewise_linear_model(input_df, X_columns, Y_column, num_breakpoints):
    train_start_time = time.time()

    # Define independent and dependent parameters 
    X = input_df[X_columns].values.flatten()  # Ensure X is a 1D array
    y = input_df[Y_column].values.flatten()

    # Split data into training and testing 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = stratified_temperature_split(X, y, num_bins = 10, test_size = 0.2, random_state=123)

    # Fit a piecewise linear model
    model = pwlf.PiecewiseLinFit(X_train, y_train)
    model.fit(num_breakpoints)  # Define number of breakpoints

    # Predict on test set
    y_test_pred = model.predict(X_test)

    train_end_time = time.time()
    training_runtime = (train_end_time - train_start_time) / 60

    return model, None, None, y_test.tolist(), y_test_pred.tolist(), training_runtime



def train_polynomial_model(input_df, X_columns, Y_column, degree, alpha):
    train_start_time = time.time()

    # Define independent and dependent parameters 
    X = input_df[X_columns]
    y = input_df[[Y_column]]

    # Split data into training and testing 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = stratified_temperature_split(X, y, num_bins = 10, test_size = 0.2, random_state=123)

    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train Ridge regression (regularized polynomial regression)
    model = Ridge(alpha=alpha)  # alpha controls regularization strength
    model.fit(X_train_poly, y_train)

    # Predict on test set
    y_test_pred = model.predict(X_test_poly)

    train_end_time = time.time()
    training_runtime = (train_end_time - train_start_time) / 60

    return model, poly, None, None, y_test.values.flatten().tolist(), y_test_pred.flatten().tolist(), training_runtime



