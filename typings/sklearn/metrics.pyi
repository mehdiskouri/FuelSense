import numpy as np

def precision_recall_fscore_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str,
    zero_division: int = ...,
) -> tuple[float, float, float, np.ndarray]: ...
