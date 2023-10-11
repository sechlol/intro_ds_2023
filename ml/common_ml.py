from typing import Dict, Any

import numpy as np


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
    correct_count = np.array([
        [t, np.sum((predictions_proba > t) == y_true)]
        for t in np.linspace(0.1, 1, 20)])
    best_i = correct_count[:, 1].argmax()
    best_accuracy = correct_count[best_i, 1] / len(predictions_proba)
    best_threshold = correct_count[best_i, 0]
    accuracy_50 = np.sum((predictions_proba > 0.5) == y_true) / len(predictions_proba)
    accuracy_dummy = np.sum(y_true == True) / len(predictions_proba)
    return {
        "best_decision_threshold": best_threshold,
        "accuracy": best_accuracy,
        "accuracy_50": accuracy_50,
        "accuracy_dummy": accuracy_dummy,
    }
