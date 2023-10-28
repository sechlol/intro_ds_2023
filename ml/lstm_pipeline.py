import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.src.callbacks import History
from sklearn import model_selection, metrics

from ml.common_ml import calculate_accuracy

# Check this useful material:
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://github.com/MohammadFneish7/Keras_LSTM_Diagram
# https://stackoverflow.com/questions/69906416/forecast-future-values-with-lstm-in-python
# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

_OUT_PATH = Path("out/lstm")
_SEED = 666
_N_STEPS = 20
_BATCHES = 64
_EPOCHS = 20


@dataclass
class LstmResultData:
    recall: float
    accuracy: Dict
    confusion_matrix: np.ndarray
    model: Sequential


def run_pipeline(data: pd.DataFrame, y_target: np.ndarray) -> Tuple[Sequential, History]:
    input_shape = (_N_STEPS, data.shape[1])

    split = model_selection.train_test_split(data.to_numpy(), y_target.flatten(), test_size=0.0001, random_state=_SEED)
    x_train, x_test, y_train, y_test = _split_train_test(split)

    model = _get_model_simple_gru(input_shape)
    history = model.fit(x_train, y_train, batch_size=_BATCHES, epochs=_EPOCHS, validation_split=0.15, shuffle=True)
    return model, history


def make_predictions(x_data: np.ndarray, y_true: np.ndarray, model: Sequential) -> LstmResultData:
    x, y = _split_sequences(np.hstack([x_data, y_true.reshape(-1, 1)]), n_steps=_N_STEPS)

    loss, binary_accuracy = model.evaluate(x, y)

    y_predict_proba = model.predict(x).flatten()
    y_predict = y_predict_proba > 0.5

    result = LstmResultData(
        model=model,
        recall=metrics.recall_score(y, y_predict),
        confusion_matrix=metrics.confusion_matrix(y, y_predict),
        accuracy={
            "binary_accuracy": binary_accuracy,
            **calculate_accuracy(y_predict_proba, y)
        })

    return result


def _get_model_simple(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(layers.LSTM(units=50, activation="tanh", input_shape=input_shape))
    model.add(layers.UnitNormalization())
    model.add(layers.Dense(units=1, activation="sigmoid"))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction="auto"),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        metrics=["binary_accuracy"])

    return model


def _get_model_simple_gru(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(layers.GRU(units=50, activation="tanh", input_shape=input_shape))
    model.add(layers.UnitNormalization())
    model.add(layers.Dense(units=1, activation="sigmoid"))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        ])

    return model


def _get_model_complex(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(layers.LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(layers.Dropout(0.2))

    # Add another LSTM layer
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.Dropout(0.2))

    # Add the final LSTM layer
    model.add(layers.LSTM(50, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def save_result(result: LstmResultData, history: History, split: Tuple):
    # Create the directory structure if it doesn't exist
    _OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Save files
    acc_json = {"recall": result.recall, **result.accuracy}
    with open(_OUT_PATH / "accuracy.json", "w") as f:
        json.dump(acc_json, f, indent=3)

    # Plot history
    for metric, values in history.history.items():
        plt.plot(history.epoch, values, label=metric)

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.xticks(history.epoch)
    plt.suptitle("LSTM training history")
    plt.savefig(_OUT_PATH / "history.png")

    x_train, x_test, y_train, y_test = _split_train_test(split)
    y_train_pred = result.model.predict(x_train) > result.accuracy["best_decision_threshold"]
    y_test_pred = result.model.predict(x_test) > result.accuracy["best_decision_threshold"]

    # Plot confusion matrix of train data
    metrics.ConfusionMatrixDisplay.from_predictions(y_train_pred, y_train.flatten())
    plt.suptitle("Train set confusion matrix")
    plt.savefig(_OUT_PATH / "confusion_train.png")

    # Plot confusion matrix of test data
    metrics.ConfusionMatrixDisplay.from_predictions(y_test_pred, y_test.flatten())
    plt.suptitle("Test set confusion matrix")
    plt.savefig(_OUT_PATH / "confusion_test.png")

    print(acc_json)
    print(f"Saved results to {_OUT_PATH}")


def _split_train_test(split: Tuple) -> Tuple:
    x_train, x_test, y_train, y_test = split
    x_train, y_train = _split_sequences(np.hstack([x_train, y_train.reshape(-1, 1)]), n_steps=_N_STEPS)
    x_test, y_test = _split_sequences(np.hstack([x_test, y_test.reshape(-1, 1)]), n_steps=_N_STEPS)
    return x_train, x_test, y_train, y_test


def _split_sequences(sequences: np.ndarray, n_steps: int):
    """
    split a multivariate sequence into samples
    """
    x, y = [], []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]

        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)
