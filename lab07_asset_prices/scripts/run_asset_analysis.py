from pathlib import Path
from lab07_asset_prices.src.asset_analysis import (
    load_and_clean_data,
    plot_closing_price,
    plot_daily_change,
    calculate_std_dev,
    calculate_daily_change,     
    save_cleaned_data
)

#paths
root_dir = Path(__file__).parent.parent
data_path = root_dir / "data" / "HistoricalData.csv"
outdir = root_dir / "artifacts"

#load and clean data
df = load_and_clean_data(data_path)
save_cleaned_data(df, outdir, company_name="Amazon")

#plot closing price
plot_closing_price(df, outdir, company_name="Amazon")

#calculate daily % change and add to dataframe
df = calculate_daily_change(df)
plot_daily_change(df, outdir, company_name="Amazon")

#calculate and print standard deviation of daily % change
std_dev = calculate_std_dev(df)
print(f"Standard Deviation of Daily % Change: {std_dev:.2f}%")

#save summary result
results_path = outdir / "results" / "summary.txt"
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, 'w') as f:
    f.write(f"Amazon Stock Volatility Report\n")
    f.write(f"==============================================\n")
    f.write(f"Standard Deviation of Daily % Change: {std_dev:.2f}%\n")
    f.write(f"Records analysed: {len(df)}\n")
    f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")