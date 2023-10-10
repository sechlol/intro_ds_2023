import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from common import data_wrangling as dw

_OUT_PATH = Path("out/xgboost")
_INDICES = ["SPY", "XLE", "XLY", "XLF", "XLV", "XLI", "XLK", "XLB", "XLU", "XLP"]
_SEED = 666


@dataclass
class Params:
    train_matrix: xgb.DMatrix
    test_matrix: xgb.DMatrix
    boost_rounds: int
    params: Dict[str, Any]
    early_stopping_rounds: int
    cv_folds: int
    verbose_training: bool = False


def run_pipeline(dataset: pd.DataFrame):
    data = preprocess_data(dataset)
    target = make_target(data, periods_difference=30)
    split = train_test_split(data.to_numpy(), target.to_numpy(), train_size=0.8, random_state=_SEED)
    x_train, x_test, y_train, y_test = split

    # Create model parameters
    p = Params(
        train_matrix=xgb.DMatrix(x_train, y_train),
        test_matrix=xgb.DMatrix(x_test, y_test),
        params={"objective": "binary:logistic", "tree_method": "hist", "eval_metric": ["error", "auc", "logloss"]},
        boost_rounds=50,
        early_stopping_rounds=20,
        cv_folds=5,
        verbose_training=False,
    )

    # Do training
    booster, history = do_training(p)

    # Process results
    predictions = booster.predict(p.test_matrix)
    accuracy, threshold = calculate_accuracy(predictions, y_test)
    contributions_raw = booster.predict(p.test_matrix, pred_contribs=True)
    contributions_df = pd.DataFrame(contributions_raw, columns=np.hstack([data.columns.values, ["bias"]]))
    extras = {"decision_threshold": threshold, "accuracy": accuracy}
    save_result(booster, history, contributions_df, extras)


def calculate_accuracy(predictions, y_test) -> Tuple[float, float]:
    """
    Calculate the best accuracy and threshold for binary classification predictions.

    Parameters:
        predictions (numpy.ndarray): Predicted binary classification probabilities.
        y_test (numpy.ndarray): True binary classification labels.

    Returns:
        Tuple[float, float]: A tuple containing the best accuracy and the corresponding threshold.

    The function calculates the accuracy for a range of thresholds and returns the threshold that
    maximizes accuracy on the provided predictions and true labels.
    """
    correct_count = np.array([
        [t, ((predictions > t) == y_test.flatten()).sum()]
        for t in np.linspace(0.1, 1, 30)])
    best_i = correct_count[:, 1].argmax()
    best_accuracy = correct_count[best_i, 1] / len(predictions)
    best_threshold = correct_count[best_i, 0]
    return best_accuracy, best_threshold


def preprocess_data(dataset: pd.DataFrame, resample_freq: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocesses a given dataset by computing Simple Moving Averages (SMAs), relative strengths,
    and optionally resampling the data to a specified frequency.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing historical data for stock indices.
        resample_freq (Optional[str]): The frequency at which to resample the data.
            Possible values are 'W' (for weekly) and 'M' (for monthly). If None, no resampling is performed.

    Returns:
        pd.DataFrame: A preprocessed DataFrame with SMAs, relative strengths, and optional resampling.
    """
    smas = dw.compute_SMA(dataset, indices=_INDICES, window_lengths=[20, 50, 200])
    rel_strengths = dw.compute_relative_strength(dataset, _INDICES)
    combined = pd.concat([dataset, rel_strengths, smas], axis=1).dropna()
    if resample_freq:
        return combined.resample(resample_freq).last()
    else:
        return combined


def make_target(dataset: pd.DataFrame, periods_difference: int) -> pd.DataFrame:
    """
    Creates a binary target variable based on the price movement of the target index.

    Parameters:
        dataset (pd.DataFrame): A DataFrame containing historical data for stock indices.
        periods_difference (int): The number of periods to look ahead for computing the target variable.
            A positive value indicates looking into the future, while a negative value looks into the past.

    Returns:
        pd.DataFrame: A DataFrame containing the binary target variable.
            - 1 indicates that the "SPY" index increased over the specified number of periods.
            - 0 indicates that the "SPY" index either remained unchanged or decreased over the specified periods.
    """
    y = dataset["SPY"] < dataset["SPY"].shift(periods=-periods_difference)
    return y.to_frame()


def save_result(booster: xgb.Booster, history: pd.DataFrame, contributions: pd.DataFrame, extras: Dict):
    """
    Save XGBoost model, feature contributions, extras dictionary, and training history.

    Parameters:
        booster (xgb.Booster): The trained XGBoost booster model.
        history (pd.DataFrame): DataFrame containing training history.
        contributions (pd.DataFrame): DataFrame containing feature contributions.
        extras (Dict): A dictionary of additional information to be saved.

    The function creates the necessary directory structure if it doesn't exist and saves
    the XGBoost model, feature contributions, extras dictionary, and training history
    as separate files in the specified output directory.
    """

    # Create the directory structure if it doesn't exist
    _OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Save files
    booster.save_model(_OUT_PATH / "booster_model.json")
    contributions.to_csv(_OUT_PATH / "features_contribution.csv")
    with open(_OUT_PATH / "extras.json", "w") as f:
        json.dump(extras, f)

    # Plot train metrics
    history.plot()
    plt.savefig(_OUT_PATH / "train_history.png")
    history.to_csv(_OUT_PATH / "train_history.csv")


def do_training(params: Params) -> Tuple[xgb.Booster, pd.DataFrame]:
    """
    Train an XGBoost model using the specified parameters.
    https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train

    Returns:
        Tuple[xgb.Booster, Dict]: A tuple containing the trained XGBoost booster model
        and a dictionary of training history.
    """
    print(f"Starting XGBoost training. {params.boost_rounds} rounds, Early stopping at {params.early_stopping_rounds}")
    history = {}
    model = xgb.train(
        params=params.params,
        dtrain=params.train_matrix,
        num_boost_round=params.boost_rounds,
        evals=[(params.test_matrix, "validation"), (params.train_matrix, "train")],
        evals_result=history,
        verbose_eval=params.boost_rounds // 20 if params.verbose_training else None,
        early_stopping_rounds=params.early_stopping_rounds)

    # Combine history into a single dataframe
    history_df = pd.concat([
        pd.DataFrame(history["train"]).add_suffix("_train"),
        pd.DataFrame(history["validation"]).add_suffix("_validation"),
    ], axis=1)

    return model, history_df


def do_cross_validation(params: Params) -> pd.DataFrame:
    """
    Perform cross-validation for an XGBoost model using the specified parameters.
    https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv

     Returns:
        pd.DataFrame: A DataFrame containing the cross-validation results.
    """
    print(f"Starting XGBoost cross-validation. {params.cv_folds}-folds, {params.boost_rounds} rounds")
    return xgb.cv(
        params=params.params,
        dtrain=params.train_matrix,
        num_boost_round=params.boost_rounds,
        nfold=params.cv_folds,
        metrics=params.params.get("eval_metric"),
        seed=_SEED)
