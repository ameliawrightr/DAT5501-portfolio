from __future__ import annotations
from typing import Union
import numpy as np
import pandas as pd

def to_np_date(value: Union[str, np.datetime64]) -> np.datetime64:
    #convert input to numpy datetime64
    if isinstance(value, np.datetime64):
        return value.astype('datetime64[D]')
    if isinstance(value, str):
        return np.datetime64(value, 'D')
    raise TypeError(f"Unsupported type: {type(value)}")

def days_since(data_input: Union[str, np.datetime64]) -> int:
    #calculate how many days have passed since the given date
    #args: data_str: str - date in the format 'YYYY-MM-DD'
    #returns: int - number of days since the given date

    today = np.datetime64('today', 'D')
    given_date = to_np_date(data_input)
    delta = today - given_date
    return int(delta.astype(int))

def load_dates_and_calculate(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    #load dates from a CSV file and calculate days since each date
    #args: csv_path: str - path to the CSV file
    #      date_col: str - name of the column containing date strings
    #returns: pd.DataFrame - original DataFrame with an additional column for days since

    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {csv_path}.")

    df['days_since'] = df[date_col].apply(days_since)
    return df