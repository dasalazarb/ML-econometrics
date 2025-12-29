"""Error metrics for evaluating LSTM predictions."""
from typing import Dict, Iterable, Tuple


def compute_errors(param_model: Dict, param_pred: Dict[str, Iterable[float]]) -> Tuple[float, float, float, float]:
    """Replicates the notebook's error aggregation logic."""
    error_5d_2 = 0
    error_10d_2 = 0
    for i in range(0, param_model["iter"]):
        error_i_5d = (param_pred["yhat_5d"][i] - param_pred["y_5d"][i]) ** 2
        error_5d_2 = error_5d_2 + error_i_5d
        error_i_10d = (param_pred["yhat_10d"][i] - param_pred["y_10d"][i]) ** 2
        error_10d_2 = error_10d_2 + error_i_10d

    error_5d_abs = 0
    error_10d_abs = 0
    for i in range(0, param_model["iter"]):
        error_i_5d = abs(param_pred["yhat_5d"][i] - param_pred["y_5d"][i])
        error_5d_abs = error_5d_abs + error_i_5d
        error_i_10d = abs(param_pred["yhat_10d"][i] - param_pred["y_10d"][i])
        error_10d_abs = error_10d_abs + error_i_10d

    return error_5d_2, error_10d_2, error_5d_abs, error_10d_abs


# Alias preserved from the notebook naming.
def medidaError(param_model: Dict, param_pred: Dict[str, Iterable[float]]) -> Tuple[float, float, float, float]:
    return compute_errors(param_model, param_pred)
