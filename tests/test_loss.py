import pytest
import numpy as np

from hungry_geese.loss import masked_mean_squared_error

@pytest.mark.parametrize('y_true, y_pred, expected_loss', [
    (np.ones((1, 1, 2)), np.ones((1, 1)), 0),
    (np.ones((1, 1, 2)), np.zeros((1, 1)), 1),
    (np.array([[[1, 1]]]), np.array([[1]]), 0),
    (np.array([[[1, 1], [1, 1]]]), np.array([[1], [1]]), 0),
    (np.array([[[1, 1], [1, 0]]]), np.array([[1], [1]]), 0),
    (np.array([[[1., 1], [2, 1]]]), np.array([[1, 1]]), 0.5),
    (np.array([[[1., 0], [2, 1]]]), np.array([[1, 1]]), 1),
    (np.array([[[1., 1], [2, 0]]]), np.array([[1, 1]]), 0),
])
def test_masked_mean_squared_error(y_true, y_pred, expected_loss):
    loss = masked_mean_squared_error(y_true, y_pred)
    assert expected_loss == loss
