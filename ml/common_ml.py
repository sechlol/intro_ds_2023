from typing import Dict, Any

import numpy as np
from sklearn import metrics


def calculate_accuracy(predictions_proba: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """
    Calculate the best accuracy and threshold for binary classification predictions.

    Parameters:
        predictions_proba (numpy.ndarray): Predicted binary classification probabilities.
        y_true (numpy.ndarray): True binary classification labels.

    Returns:
        Tuple[float, float]: A tuple containing the best accuracy and the corresponding threshold.

    The function calculates the accuracy for a range of thresholds and returns the threshold that
    maximizes accuracy on the provided predictions and true labels.
    """
    y_true = y_true.flatten()
    accuracy_50 = np.sum((predictions_proba > 0.5) == y_true) / len(predictions_proba)
    accuracy_dummy = np.sum(y_true == True) / len(predictions_proba)
    y_pred_best = (predictions_proba > 0.5).astype(float)
    return {
        "accuracy": accuracy_50,
        "accuracy_dummy": accuracy_dummy,
        "f1_score": metrics.f1_score(y_true, y_pred_best),
        "recall": metrics.recall_score(y_true.astype(int), y_pred_best.astype(int)),
        "precision": metrics.precision_score(y_true, y_pred_best, zero_division=0),
        "r2_score": metrics.r2_score(y_true, y_pred_best),
    }
