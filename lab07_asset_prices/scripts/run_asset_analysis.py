import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from lab07_asset_prices.src.asset_analysis import (
    load_and_clean_data,
    plot_closing_price,
    plot_daily_change,
    calculate_std_dev,
    calculate_daily_change,
)
from lab07_asset_prices.src.asset_sorting_activity import (
    run_end_to_end as run_sorting_analysis,
)

#paths
pkg_root = Path(__file__).resolve().parents[1]
outdir = pkg_root/"artifacts"
data_path = pkg_root/"data"/"HistoricalData.csv"


#load and clean data
df = load_and_clean_data(data_path)

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
with open(results_path, 'w') as f:
    f.write(f"Amazon Stock Volatility Report\n")
    f.write(f"Standard Deviation of Daily % Change: {std_dev:.2f}%\n")
    f.write(f"Records analysed: {len(df)}\n")
    f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")


#call sorting task
print("\nRunning sorting analysis...")
run_sorting_analysis(data_path, outdir)
print("Sorting analysis completed.")

