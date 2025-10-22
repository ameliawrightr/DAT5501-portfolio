from lab07_data_fitting.src.data_fitting import linear, fit_model, correlation_coefficient
import numpy as np

def test_linear_fit():
    x = np.arange(0, 10)
    y = 3 * x + 2
    y_fit, params = fit_model(x, y, model="linear")
    assert abs(params[0] - 3) < 1e-6
    assert abs(params[1] - 2) < 1e-6

def test_correlation():
    x = np.linspace(0, 1, 5)
    y = x.copy()
    r = correlation_coefficient(x, y)
    assert abs(r - 1) < 1e-6

