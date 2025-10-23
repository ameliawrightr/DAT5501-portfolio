# Lab 07 — Data Fitting Extensions (US Election Dataset)

**Branch:** `lab7part4-data-fitting-extensions`  
**Folder:** `lab07_data_fitting/`

This lab extends your previous US election analysis with **model fitting and evaluation**.  
You implement **OLS** and **Maximum Likelihood (MLE)** fits, compare **linear / quadratic / exponential** models, and evaluate them using **χ², reduced-χ², AIC, BIC**. Residual diagnostics and plots are saved to `/outputs`.

---

## Objectives
- Fit linear, quadratic, and exponential models to real data.
- Compute correlation (r), χ², reduced-χ², AIC, BIC.
- Do **MLE (Gaussian errors)** to estimate parameters **and** noise σ̂, and compute likelihood-based AIC/BIC.
- Produce and interpret plots (fit lines + residuals).
- Summarise and compare models; justify a best choice.

---

## Data
`lab07_data_fitting/data/us_election.csv` (semicolon `;` delimited)

Columns used:
- `votes` — raw vote count for a candidate in a county (x)
- `fraction_votes` — fraction of that county’s votes won by that candidate (y, coerced to `[0,1]`, `%` handled)

**Note:** Each row is **candidate–county**; a county appears multiple times (one row per candidate).

---

## Environment & Dependencies
Activate your venv and install:
```bash
pip install -U numpy pandas matplotlib scipy

python3 -m lab07_data_fitting.scripts.run_data_fitting

