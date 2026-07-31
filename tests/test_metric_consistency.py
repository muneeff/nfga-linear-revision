import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def test_rmse_r2_ordering_on_common_target():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    good = np.array([0.0, 1.1, 1.9, 3.0])
    bad = np.array([0.0, 2.0, 0.0, 4.0])
    rmse_good = mean_squared_error(y, good) ** 0.5
    rmse_bad = mean_squared_error(y, bad) ** 0.5
    assert rmse_good < rmse_bad
    assert r2_score(y, good) > r2_score(y, bad)
