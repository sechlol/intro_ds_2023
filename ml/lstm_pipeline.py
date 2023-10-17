from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.src.utils import timeseries_dataset_from_array
from sklearn import model_selection

_SEED = 666


@dataclass
class LstmResultData:
    model: Sequential


def run_pipeline(data: pd.DataFrame, y_target: np.ndarray) -> LstmResultData:
    x_data = data.to_numpy()
    #x_data = x_data.reshape(x_data.shape[0], 4418, 108)
    y_target = y_target.flatten()
    split = model_selection.train_test_split(x_data, y_target, test_size=0.2, random_state=_SEED)
    x_train, x_test, y_train, y_test = split

    model = Sequential()
    model.add(LSTM(units=50, activation="relu", batch_input_shape=(1, 64, 10)))  # Add LSTM layer with input shape
    model.add(Dense(units=1, activation="sigmoid"))  # Output layer with 'n' units for binary predictions

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lol = model.fit(x_train, y_train, batch_size=128, epochs=5)
    loss, accuracy = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)

    return LstmResultData(model)


def make_predictions(x_data: np.ndarray, y_true: np.ndarray, model) -> Dict[str, Any]:
    return {}
