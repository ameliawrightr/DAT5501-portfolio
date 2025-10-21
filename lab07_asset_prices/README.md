# Lab 07 — Asset Prices (Amazon 1-Year)

```bash
# Objective:
# Analyse one year of Amazon (AMZN) stock prices.
# - Load and clean CSV
# - Plot closing price over time
# - Compute daily % change (returns)
# - Plot daily % change
# - Calculate standard deviation (volatility)
# - Save all outputs to artifacts/

# Run full pipeline from repo root
python -m lab07_asset_prices.scripts.run_asset_analysis


## Extension ideas
# 1. Add rolling 30-day volatility
# 2. Compare AMZN with NASDAQ index
# 3. Add CLI flags for input/output/ticker
# 4. Timestamped output folders for experiment tracking
