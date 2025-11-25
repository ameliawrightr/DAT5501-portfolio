# Lab 08 – Fitting & Forecasting (World Population)

This lab fits and compares models for world population data (1950–2023), then uses them to forecast future population.

## Structure

```text
lab08_fitting_and_forecasting/
├── data/
│   └── worldpopulation.csv
├── src/
│   └── world_population.py
└── artifacts/
    ├── PLOT1_world_population_over_time_full_series.png
    ├── PLOT2_world_population_polynomial_fits.png
    ├── PLOT3_world_population_polynomial_forecasts.png
    ├── PLOT4_chi2_per_dof_vs_polynomial_degree.png
    ├── PLOT5_BIC_vs_polynomial_degree.png
    └── PLOT6_best_polynomial_vs_exponential_fit.png
````

## How to run

```bash
cd /Users/amelia/DAT5501-portfolio
source .venv/bin/activate
python lab08_fitting_and_forecasting/src/world_population.py
```

Requirements: `numpy`, `pandas`, `matplotlib`, `scipy`.

## What the code does

1. **Data prep & exploration**

   * Reads `worldpopulation.csv`
   * Filters to `Entity == "World"`
   * Plots population vs year (PLOT1).

2. **Train/test split & polynomial fits**

   * Training: all years up to last year − 10
   * Test: last 10 years
   * Fits polynomials of degree 1–9 on training data.
   * Plots fits on history (PLOT2) and 10-year forecasts (PLOT3).

3. **Model comparison (χ² & BIC)**

   * Computes χ², χ²/dof, and BIC for each degree.
   * Plots χ²/dof vs degree (PLOT4) and BIC vs degree (PLOT5).
   * Chooses **best polynomial** as the one with minimum BIC (here: degree 7).

4. **Parameter values & uncertainties (best model)**

   * Builds design matrix and solves least squares.
   * Computes covariance matrix of coefficients and 1-σ uncertainties.

5. **Exponential alternative**

   * Fits ( y = A \exp(k (x - x_0)) ) with `curve_fit`.
   * Gets parameter errors and BIC.
   * Compares best polynomial BIC vs exponential BIC.
   * Plots best polynomial vs exponential fit (PLOT6).

## Key points

* χ²/dof decreases with degree, but BIC penalises overly complex polynomials.
* A moderate-degree polynomial (selected by BIC) balances fit and complexity.
* The exponential model provides a simple, physically reasonable alternative and is judged against the polynomial using BIC.