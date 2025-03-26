import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Расчет точности.
    """
    return np.mean(y_true == y_pred)