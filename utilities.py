from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_model(Sequential, sequence_length, units=256,
                 cell=LSTM, n_layers=2,
                 dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop",
                 bidirectional=False):
    """
    Creates and returns Machine Learning model based on input parameters.
    Many of these parameters you do not need to set themselves but all are
    described below.

    Parameters (type):
    -----------
        Sequntial (function): Function to base model of off
        sequence_length (int): The length of the input sequence
        units (int):
        cell (tensorflow.keras.layer): The type of cell to use in the Nueral
        Network, default is LSTM a recurrent nueral network
        n_layers (int): Number of layers in the neural network
        dropout (float): 
        loss (string): Type of loss function to use
        optimizer (string): Type of optimizer to use
        bidirectional (bool): Whether network should be directed (False) or
        bidirected (True)

    Returns:
    -------
        Machine Learning model
    """
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                                        input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True,
                               input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)
    return model


def predict(model, data, N_STEPS):
    """
    Uses trained model to predict and return tomorrows stock price.

    Parameters (type):
    -----------------
        model (tensorflow model): Trained tensorflow model
        data : Data used to train model
        N_STEPS (int): Number of observations to use in order to predict stock
        price

    Returns:
    -------
        predicted_price (float): The predicted stock price
    """
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape(
        (last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[
        0][0]
    return predicted_price


def plot_graph(model, data):
    """
    Plots the data and the predicted price
    """
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
        np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]
                        ["adjclose"].inverse_transform(y_pred))
    # last 200 days, feel free to edit that
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_accuracy(model, data, LOOKUP_STEP):
    """
    Calculates the accuracy of the model
    """
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
        np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]
                        ["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(
        current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(
        current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
              test_size=0.2, feature_columns=['adjclose', 'volume', 'open',
                                              'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling,
    normalizing and splitting.

    Parameters (type):
    -----------------
        ticker (str/pd.DataFrame): the ticker you want to load, examples
        include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used
        to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1
        (e.g next day)
        test_size (float): ratio for test data, default is 0.2
        (20% testing data)
        feature_columns (list): the list of features to use to feed into the
        model, default is everything grabbed from yahoo_fin
    """

    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError(
            "ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with
    # `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be
    # of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not
    # available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network

    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"],
    result["y_test"] = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle)
    # return the result
    return result
