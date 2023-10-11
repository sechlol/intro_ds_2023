import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

_OUT_PATH = Path("out/xgboost")
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


def run_pipeline(data: pd.DataFrame, y_target: np.ndarray, cross_validate: bool) -> ResultData:
    x_data = data.to_numpy()
    split = train_test_split(x_data, y_target, train_size=0.8, random_state=_SEED)
    x_train, x_test, y_train, y_test = split

    # Create model parameters
    p = Params(
        train_matrix=xgb.DMatrix(x_train, y_train),
        test_matrix=xgb.DMatrix(x_test, y_test),
        params={
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 3,
            "min_child_weight": 4,
            # "alpha": 20,  # L1 regularization
            # "lambda": 2,  # L2 regularization
            # "gamma": 50,
            "eta": 0.1,
            "eval_metric": ["error", "auc", "logloss"]},
        boost_rounds=500,
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


def make_predictions(x_data: np.ndarray, y_true: np.ndarray, model: xgb.Booster):
    x_matrix = xgb.DMatrix(x_data)
    predictions = model.predict(x_matrix)
    return _calculate_accuracy(predictions, y_true)


def _calculate_accuracy(predictions: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
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
