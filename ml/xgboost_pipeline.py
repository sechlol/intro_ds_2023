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


@dataclass
class ResultData:
    booster: xgb.Booster
    history: pd.DataFrame
    contributions: pd.DataFrame
    extras: Dict
    cv_history: Optional[pd.DataFrame] = None


def run_pipeline(dataset: pd.DataFrame, cross_validate: bool) -> ResultData:
    data = _preprocess_data(dataset)
    target = _make_target(data, periods_difference=30)
    split = train_test_split(data.to_numpy(), target.to_numpy(), train_size=0.95, random_state=_SEED)
    x_train, x_test, y_train, y_test = split

    # Create model parameters
    p = Params(
        train_matrix=xgb.DMatrix(x_train, y_train),
        test_matrix=xgb.DMatrix(x_test, y_test),
        params={
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 10,
            "min_child_weight": 5,
            "alpha": 10,  # L1 regularization
            "lambda": 10,  # L2 regularization
            "gamma": 50,
            "eta": 0.01,
            "eval_metric": ["error", "auc", "logloss"]},
        boost_rounds=50,
        early_stopping_rounds=20,
        cv_folds=5,
        verbose_training=False,
    )

    # Do training
    cv_result = _do_cross_validation(p) if cross_validate else None
    booster, history = _do_training(p)

    # Process results
    predict_matrix = xgb.DMatrix(x_test)
    predictions = booster.predict(predict_matrix)
    accuracy_data = _calculate_accuracy(predictions, y_test)
    contributions_raw = booster.predict(predict_matrix, pred_contribs=True)
    contributions_df = pd.DataFrame(contributions_raw, columns=np.hstack([data.columns.values, ["bias"]]))

    result_data = ResultData(
        booster=booster,
        history=history,
        contributions=contributions_df,
        extras=accuracy_data,
        cv_history=cv_result)

    _save_result(result_data)
    return result_data


def make_predictions(data: pd.DataFrame, model: xgb.Booster):
    x_data = _preprocess_data(data)
    y_true = _make_target(x_data, periods_difference=30)
    x_data = x_data["2022":]
    y_true = y_true["2022":]
    y_arr = y_true.to_numpy()
    x_matrix = xgb.DMatrix(x_data.to_numpy())

    predictions = model.predict(x_matrix)
    accuracy_data = _calculate_accuracy(predictions, y_arr)
    spy_30 = x_data["SPY"].shift(-30)
    grr = pd.DataFrame({
        "y_true": y_arr.flatten(),
        "prob": predictions.flatten(),
        "y_pred": predictions.flatten() > 0.5,
        "SPY": x_data["SPY"],
        "SPY30": spy_30.to_numpy(),
        "CONTROL": x_data["SPY"] < spy_30,
    }, index=x_data.index)
    print(accuracy_data)


def _calculate_accuracy(predictions, y_test) -> Dict[str, Any]:
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
    y_test = y_test.flatten()
    correct_count = np.array([
        [t, np.sum((predictions > t) == y_test)]
        for t in np.linspace(0.1, 1, 20)])
    best_i = correct_count[:, 1].argmax()
    best_accuracy = correct_count[best_i, 1] / len(predictions)
    best_threshold = correct_count[best_i, 0]
    accuracy_50 = np.sum((predictions > 0.5) == y_test) / len(predictions)
    accuracy_dummy = np.sum(y_test == True) / len(predictions)
    return {
        "best_decision_threshold": best_threshold,
        "accuracy": best_accuracy,
        "accuracy_50": accuracy_50,
        "accuracy_dummy": accuracy_dummy,
    }


def _preprocess_data(dataset: pd.DataFrame, resample_freq: Optional[str] = None) -> pd.DataFrame:
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


def _make_target(dataset: pd.DataFrame, periods_difference: int) -> pd.DataFrame:
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


def _save_result(result: ResultData):
    """
    Save XGBoost model, feature contributions, extras dictionary, and training history.

    Parameters:
        result (ResultData): An instance of ResultData containing result data.

    The function creates the necessary directory structure if it doesn't exist and saves
    the XGBoost model, feature contributions, extras dictionary, and training history
    as separate files in the specified output directory.
    """

    # Create the directory structure if it doesn't exist
    _OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Save files
    result.history.to_csv(_OUT_PATH / "train_history.csv")
    result.booster.save_model(_OUT_PATH / "booster_model.json")
    result.contributions.to_csv(_OUT_PATH / "features_contribution.csv")
    with open(_OUT_PATH / "extras.json", "w") as f:
        json.dump(result.extras, f)

    # Plot train metrics
    result.history.plot()
    plt.xlabel("Training steps")
    plt.ylabel("Metrics values")
    plt.suptitle("XGBoost model convergence")
    plt.tight_layout()
    plt.savefig(_OUT_PATH / "train_history.png")

    # Plot cross validation metrics
    if result.cv_history is not None:
        result.cv_history.plot()
        plt.xlabel("Training steps")
        plt.ylabel("CV Metrics values")
        plt.suptitle("XGBoost Cross-Validation")
        plt.tight_layout()
        plt.savefig(_OUT_PATH / "cv_history.png")

    print(result.extras)
    print(f"Saved results to {_OUT_PATH}")


def _do_training(params: Params) -> Tuple[xgb.Booster, pd.DataFrame]:
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


def _do_cross_validation(params: Params) -> pd.DataFrame:
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
