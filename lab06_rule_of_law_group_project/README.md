# Lab 06 – Rule of Law Group Project

This lab analyses a Rule of Law–style index across countries and years, exploring trends and fitting simple models to describe and forecast changes over time.

## Structure

```text
lab06_rule_of_law_group_project/
├── data/
│   └── rule_of_law.csv
├── src/
│   └── rule_of_law_analysis.py
└── artifacts/
    ├── fig1_germany_1930_1950.png
    ├── fig2_russia_1999_2024.png
    ├── fig3_since_regime_start_topbaseline.png
    ├── fig4_grouped_pre_vs_war.png
    └──
    fig4a_dual_axis.png
    
## How to run

    cd /Users/amelia/DAT5501-portfolio
    source .venv/bin/activate
    python lab06_rule_of_law_group_project/src/rule_of_law_analysis.py

## Requirements

- numpy
- pandas
- matplotlib
- scipy (or statsmodels if you use it)
- seaborn (if used for plots)

## What the code does

### 1. Data loading & cleaning

- Reads `rule_of_law.csv`.
- Ensures consistent column types (`country`, `year`, `score`, plus any `region` / `income_group` variables).
- Drops or flags rows with missing/invalid values (e.g. impossible scores).
- Optionally normalises scores (e.g. 0–100 → 0–1).
- Writes a cleaned DataFrame to `artifacts/TABLE1_country_summary_statistics.csv` (or to `data/processed/` if you use that pattern).

### 2. Descriptive analysis & plots

- Computes basic descriptive stats by **country** and **year** (mean, min, max, count).
- Plots the overall score distribution for the latest year (PLOT1).
- Plots time series of scores for a selected subset of countries (PLOT2).
- If a `region` variable exists, computes regional averages and plots regional trends over time (PLOT3).

### 3. Trend modelling (per country)

For each country with enough data points:

- Fits a simple linear model:  
  `score = β0 + β1 * year + ε`
- Stores `β0`, `β1`, `R²`, number of observations, and possibly p-values in `TABLE2_country_trend_coefficients.csv`.
- Plots example country fits (historical data + fitted line) for a few representative cases (improving / declining / flat) in PLOT4.

### 4. Forecasts

- Using the fitted linear model for each country, extrapolates scores forward to a small set of future years (e.g. +5 or +10 years).
- Stores forecast values in `TABLE3_country_forecasts.csv`.
- Optionally overlays historical data and forecasted points for selected countries.

### 5. Basic interpretation (printed to console / comments)

Prints summary info to the console, for example:

- Number of countries with significantly positive vs negative trends.
- Countries with the strongest improvements or declines.

Comments in the code describe how to interpret `β1` (slope) as **average yearly change in rule-of-law score**.

## Key points

- The dataset is a **panel**: multiple countries observed repeatedly over time.
- Simple linear models give a rough view of whether a country is **improving, stable, or deteriorating** in rule-of-law, but:
  - They ignore nonlinear effects, political shocks, and structural breaks.
- Comparing trends across countries or regions highlights **relative progress** and **divergence**.
- Forecasts based on linear trends are **illustrative only** – useful to understand the model, not to make serious policy predictions.

From a data-science perspective, this lab exercises:

- Cleaning and reshaping panel data.
- Systematic plotting and saving of artifacts for a report.
- Fitting and interpreting a large number of simple models in a loop.
