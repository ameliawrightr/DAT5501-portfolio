# DAT5501 Portfolio 
### Amelia Wright Rocafort 230002276

This repository organises all labs for the **DAT5501** module in **one clean place**.
Each lab focuses on a core industry relevant data engineering skill - inclusive of Git worflows, CI pipelines, and automated testing.

## 📂 Repository Overview
- `lab01_version_control/` - Git fundamentals – commits, branching, merging, version history 
- `lab02_unit_testing/` - Test-Driven Development (TDD), `pytest`, writing and running automated tests 
-`lab03_continuous_integration/` - CircleCI setup, automated builds, workflow configuration 
- `lab04_data_pipeline/` - Modular Python scripts, reproducible results, CLI execution, JSON configs 
- `lab05_calendar_printer/` - Programmatic date formatting, string layout logic, unit testing edge cases 
- `lab06_rule_of_law_group_project/` – Rule of law index analysis and forecasting (group project).
- `lab07_asset_prices/` – Financial time-series analysis of asset prices.
- `lab07_data_fitting/` – Curve fitting and model comparison on real data.
- `lab08_fitting_and_forecasting/` – Polynomial fitting and world-population forecasting.
- `lab09_decision_tree/` – Decision-tree classification and model tuning.

Each lab is **self-contained**, following a consistent folder structure: 
labXX_name/
├── src/ # Python source code
├── tests/ # pytest-based unit tests
├── requirements.txt # Dependencies (if any)
└── README.md # Lab-specific explanation

## ⚙️ Continuous Integration (CI)
The repository integrates with **CircleCI** to automatically test all labs.

**How it works:**
- CircleCI scans for any `lab*/tests/` folders.
- If a lab includes a `requirements.txt`, dependencies are installed first.
- `pytest` is executed for each lab sequentially to validate all implementations.

✅ This ensures that every lab remains **modular**, **independently verifiable**, and **CI-compatible**.


## 💻 Local Quickstart

If you want to run all labs locally:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Upgrade pip
pip install -U pip

# 3. Install dependencies and run all tests
for d in lab*/; do
  [ -d "$d/tests" ] && pip install -r "$d/requirements.txt" 2>/dev/null || true
  [ -d "$d/tests" ] && pytest -q "$d/tests"

# 4. Run with 
python -m labXX_name.src.<main_script>

done
```

## ✨ Author
### Amelia Wright Rocafort
BSc (Hons) Digital & Technology Solutions (Data Analytics Pathway)
Queen Mary University of London