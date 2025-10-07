# DAT5501 Portfolio 
### Amelia Wright Rocafort 230002276

This repository organises all labs for the **DAT5501** module in **one clean place**.
Each lab focuses on a core industry relevant data engineering skill - inclusive of Git worflows, CI pipelines, and automated testing.

## 📂 Repository Overview
| Lab | Folder | Focus | Key Concepts |
|-----|---------|--------|--------------|
| **Lab 01** | `lab01_version_control/` | **Version Control** | Git fundamentals – commits, branching, merging, version history |
| **Lab 02** | `lab02_unit_testing/` | **Unit Testing** | Test-Driven Development (TDD), `pytest`, writing and running automated tests |
| **Lab 03** | `lab03_continuous_integration/` | **Continuous Integration (CI)** | CircleCI setup, automated builds, workflow configuration |
| **Lab 04** | `lab04_data_pipeline/` | **Data Pipeline Development** | Modular Python scripts, reproducible results, CLI execution, JSON configs |
| **Lab 05** | `lab05_calendar_printer/` | **Calendar Printer Utility** | Programmatic date formatting, string layout logic, unit testing edge cases |

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
done
```

## ✨ Author
### Amelia Wright Rocafort
BSc (Hons) Digital & Technology Solutions (Data Analytics Pathway)
Queen Mary University of London