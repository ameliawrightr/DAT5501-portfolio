# DAT5501 Portfolio 
## Amelia Wright Rocafort 230002276

This repository organises all labs for DAT5501 in **one clean place**.

## Structure
- `lab01_version_control/` – Version Control Activity (Git basics: repo, commits, branches, merge)
- `lab02_unit_testing/` – Unit Testing Activity (pytest, TDD mini-task)
- `lab03_continuous_integration/` – Continuous Integration Activity (CircleCI)

Each lab is **self-contained** with its own `README.md`, `src/`, `tests/`, and `requirements.txt` so CI can run them independently.

## How CI works
- CircleCI looks for any `lab*/tests` and runs `pytest` in that folder.
- If a lab has a `requirements.txt`, it is installed prior to running tests.

### Local quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
# run all labs' tests
for d in lab*/; do
  [ -d "$d/tests" ] && pip install -r "$d/requirements.txt" 2>/dev/null || true
  [ -d "$d/tests" ] && pytest -q "$d/tests"
done
```