Thu Oct  2 11:55:21 BST 2025
Thu Oct  2 11:58:08 BST 2025
Thu Oct  2 13:33:30 BST 2025
Thu Oct  2 13:35:19 BST 2025

# lab04_data_pipeline

## Overview
This lab demonstrates a reproducible **data pipeline** that:

1. **Generates synthetic data** from a straight line `y = m*x + b` with Gaussian noise.  
2. **Saves the dataset** as a CSV file (plus a JSON file with ground-truth parameters).  
3. **Fits a straight line** to the data using least-squares regression.  
4. **Saves a plot** showing the noisy data, the true line, and the best-fit line.  
5. **Runs unit tests** to validate that the pipeline produces correct and reproducible outputs.  
6. **Runs automatically in CI** (CircleCI), with artifacts (CSV, plot, results JSON) stored for inspection.

## What this does
- Generate synthetic (x, y) from a true line y = m*x + b with Gaussian noise → **CSV**
- Fit a straight line via least-squares → estimate **m**, **b**
- Save a **plot** showing data + true line + best-fit line
- Unit tests for CSV validity, fit quality, and plot creation
- Save RMSE/R² in `outputs/results.json`

## Quickstart
```bash
source .venv/bin/activate
python -m pip install -r lab04_data_pipeline/requirements.txt
python -m lab04_data_pipeline.scripts.run_pipeline
pytest -q lab04_data_pipeline/tests

Fri Oct  3 17:24:51 BST 2025
Fri Oct  3 17:45:17 BST 2025
Fri Oct  3 17:48:00 BST 2025


## Run commands
# Linear
python -m lab04_data_pipeline.scripts.run_pipeline --model line --n 250 --seed 1 --outdir artifacts/line

# Quadratic
python -m lab04_data_pipeline.scripts.run_pipeline \
  --model quadratic --quad-a 0.3 --quad-b -1.2 --quad-c 2.0 \
  --n 400 --seed 11 --x-start -2 --x-stop 2 --outdir artifacts/quad

# Exponential
python -m lab04_data_pipeline.scripts.run_pipeline \
  --model exponential --m 1.8 --b 0.6 --sigma-ln 0.15 \
  --n 500 --seed 22 --x-start 0 --x-stop 4 --outdir artifacts/exp

#Folder structure
- artifacts contains generated plots and csv files the plots are written from 
- data contains the original csv data file
- outputs contains the main line fit plot
- scripts contains the run file
- src contains the functions
- tests contains the test

There is also a requirements.txt file included and personal_notes.txt to be ignored. 