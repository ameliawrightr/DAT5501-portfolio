import numpy as np
from lab07_date_duration_calculator.src.duration_calc import days_since

def test_days_since_today():
    today = str(np.datetime64('today', 'D'))
    assert days_since(today) == 0

def test_days_since_past_date():
    past_date = str(np.datetime64('2000-01-01', 'D'))
    expected_days = (np.datetime64('today', 'D') - np.datetime64('2000-01-01', 'D')).astype(int)
    assert days_since(past_date) == expected_days

def test_days_since_future_date():
    future_date = str(np.datetime64('3000-01-01', 'D'))
    expected_days = (np.datetime64('today', 'D') - np.datetime64('3000-01-01', 'D')).astype(int)
    assert days_since(future_date) == expected_days

    