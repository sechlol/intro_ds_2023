import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from ml.common_ml import calculate_accuracy

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
class XGBResultData:
    booster: xgb.Booster
    history: pd.DataFrame
    contributions: pd.DataFrame
    train_scores: Dict
    cv_history: Optional[pd.DataFrame] = None


def run_pipeline(data: pd.DataFrame, y_target: np.ndarray, cross_validate: bool) -> XGBResultData:
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
    accuracy_data = calculate_accuracy(predictions, y_test)
    contributions_raw = booster.predict(predict_matrix, pred_contribs=True)
    contributions_df = pd.DataFrame(contributions_raw, columns=np.hstack([data.columns.values, ["bias"]]))

    result_data = XGBResultData(
        booster=booster,
        history=history,
        contributions=contributions_df,
        train_scores=accuracy_data,
        cv_history=cv_result)

    _save_result(result_data, split)
    return result_data


def make_predictions(x_data: np.ndarray, y_true: np.ndarray, model: xgb.Booster):
    x_matrix = xgb.DMatrix(x_data)
    predictions = model.predict(x_matrix)
    accuracy_score = calculate_accuracy(predictions, y_true)

    print("Test Accuracy:\n", accuracy_score)
    with open(_OUT_PATH / "accuracy_test.json", "w") as f:
        json.dump(accuracy_score, f, indent=3)

    return accuracy_score


def _save_result(result: XGBResultData, split):
    """
    Save XGBoost model, feature contributions, extras dictionary, and training history.

    Parameters:
        result (XGBResultData): An instance of ResultData containing result data.

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
    with open(_OUT_PATH / "accuracy_train.json", "w") as f:
        json.dump(result.train_scores, f, indent=3)

    # Plot train metrics
    result.history.plot()
    plt.xlabel("Training steps")
    plt.ylabel("Metrics values")
    plt.suptitle("XGBoost model convergence")
    plt.tight_layout()
    plt.savefig(_OUT_PATH / "xgb_train_history.png")

    # Plot cross validation metrics
    if result.cv_history is not None:
        result.cv_history.plot()
        plt.xlabel("Training steps")
        plt.ylabel("CV Metrics values")
        plt.suptitle("XGBoost Cross-Validation")
        plt.tight_layout()
        plt.savefig(_OUT_PATH / "xgb_cv_history.png")

    x_train, x_test, y_train, y_test = split
    y_train_pred = result.booster.predict(xgb.DMatrix(x_train)) > 0.5
    y_test_pred = result.booster.predict(xgb.DMatrix(x_test)) > 0.5

    # Plot confusion matrix of train data
    metrics.ConfusionMatrixDisplay.from_predictions(y_train_pred, y_train.flatten())
    plt.suptitle("Train set confusion matrix")
    plt.savefig(_OUT_PATH / "xgb_confusion_train.png")

    # Plot confusion matrix of test data
    metrics.ConfusionMatrixDisplay.from_predictions(y_test_pred, y_test.flatten())
    plt.suptitle("Test set confusion matrix")
    plt.savefig(_OUT_PATH / "xgb_confusion_test.png")

    print("Train Accuracy:\n", result.train_scores)


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
