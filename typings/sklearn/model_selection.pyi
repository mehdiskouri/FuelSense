import numpy as np

def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    stratify: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
