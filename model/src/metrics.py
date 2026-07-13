import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return {
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    denominator = matrix["true_positive"] + matrix["false_positive"]

    if denominator == 0:
        return 0.0

    return matrix["true_positive"] / denominator


def recall(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    denominator = matrix["true_positive"] + matrix["false_negative"]

    if denominator == 0:
        return 0.0

    return matrix["true_positive"] / denominator


def f1_score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    denominator = precision_value + recall_value

    if denominator == 0:
        return 0.0

    return 2 * precision_value * recall_value / denominator
