import numpy as np

from piadaptwin.gauges.regression import horizon_mae, mae, rmse


def test_mae_rmse_match_numpy() -> None:
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.5, 1.5, 2.0, 5.0])
    assert abs(mae(pred, target) - float(np.mean(np.abs(pred - target)))) < 1e-9
    assert abs(rmse(pred, target) - float(np.sqrt(np.mean((pred - target) ** 2)))) < 1e-9


def test_horizon_mae_per_column() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 0.0], [0.0, 4.0]])
    values = horizon_mae(pred, target)
    assert abs(values[0] - 1.5) < 1e-9
    assert abs(values[1] - 1.0) < 1e-9
