import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)


def distances_to_probabilities(distances):

    # Smaller distance = higher probability.
    logits = -distances

    logits = (
        logits
        - logits.max(axis=1, keepdims=True)
    )

    exp_logits = np.exp(logits)

    probabilities = (
        exp_logits
        / exp_logits.sum(
            axis=1,
            keepdims=True
        )
    )

    return probabilities


def evaluate_predictions(
    y_true,
    y_pred,
    distances
):

    probabilities = distances_to_probabilities(
        distances
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    auc_score = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc_score,
        "confusion_matrix": cm,
        "probabilities": probabilities
    }