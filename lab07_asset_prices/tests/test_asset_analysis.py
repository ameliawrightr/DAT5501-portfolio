from lab07_asset_prices.src.asset_analysis import (
    load_and_clean_data, 
    calculate_daily_change
)
from pathlib import Path

def test_load_data():
    df = load_and_clean_data(Path("lab07_asset_prices/data/HistoricalData.csv"))
    assert not df.empty
    assert "Close/Last" in df.columns

def test_daily_change():
    df = load_and_clean_data(Path("lab07_asset_prices/data/HistoricalData.csv"))
    df = calculate_daily_change(df)
    assert "Daily % Change" in df.columns
