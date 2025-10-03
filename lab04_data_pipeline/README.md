Thu Oct  2 11:55:21 BST 2025
Thu Oct  2 11:58:08 BST 2025
Thu Oct  2 13:33:30 BST 2025
Thu Oct  2 13:35:19 BST 2025

# lab04_data_pipeline

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
