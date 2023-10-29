import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neural_network, model_selection, metrics

from ml.common_ml import calculate_accuracy

_OUT_PATH = Path("out/mlp")
_SEED = 666


@dataclass
class MlpResultData:
    model: neural_network.MLPClassifier
    recall: float
    train_accuracy: Dict
    confusion_matrix: np.ndarray
    cv_scores: Optional[pd.DataFrame] = None


def run_pipeline(data: pd.DataFrame, y_target: np.ndarray, cross_validate: bool) -> MlpResultData:
    x_data = data.to_numpy()
    y_target = y_target.flatten()
    split = model_selection.train_test_split(x_data, y_target, test_size=0.2, random_state=_SEED)
    x_train, x_test, y_train, y_test = split

    model = neural_network.MLPClassifier(
        hidden_layer_sizes=100,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.01,
        n_iter_no_change=20,
        shuffle=True,
        # learning_rate="constant",
        solver="sgd",
        batch_size=32,
        random_state=_SEED,
    ).fit(x_train, y_train)

    y_predict_proba = model.predict_proba(x_test)
    y_predict = y_predict_proba[:, 1] > 0.5

    result = MlpResultData(
        model=model,
        recall=metrics.recall_score(y_test, y_predict),
        confusion_matrix=metrics.confusion_matrix(y_test, y_predict),
        train_accuracy=calculate_accuracy(y_predict_proba[:, 1], y_test))

    if cross_validate:
        scores = ["accuracy", "average_precision", "f1", "roc_auc"]
        result.cv_scores = model_selection.cross_validate(model, x_train, y_train, cv=5, scoring=scores)

    _save_results(result, model, split)
    return result


def make_predictions(x_data: np.ndarray, y_true: np.ndarray, model: neural_network.MLPClassifier) -> Dict[str, Any]:
    predictions = model.predict_proba(x_data)
    accuracy_scores = calculate_accuracy(predictions[:, 1], y_true)

    print("Test Accuracy:\n", accuracy_scores)
    with open(_OUT_PATH / "accuracy_test.json", "w") as f:
        json.dump(accuracy_scores, f, indent=3)
    return accuracy_scores


def _save_results(result: MlpResultData, model, split):
    x_train, x_test, y_train, y_test = split

    _OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix of train data
    metrics.ConfusionMatrixDisplay.from_estimator(model, x_train, y_train)
    plt.suptitle("Train set confusion matrix")
    plt.savefig(_OUT_PATH / "confusion_train.png")

    # Plot confusion matrix of test data
    metrics.ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.suptitle("Test set confusion matrix")
    plt.savefig(_OUT_PATH / "confusion_test.png")

    with open(_OUT_PATH / "accuracy_train.json", "w") as f:
        json.dump(result.train_accuracy, f, indent=3)

    print("Train Accuracy:\n", result.train_accuracy)
