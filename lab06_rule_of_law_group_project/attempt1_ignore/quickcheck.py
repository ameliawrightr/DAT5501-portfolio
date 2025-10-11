# file: lab06_rule_of_law_group_project/src/check_columns.py
from pathlib import Path
import pandas as pd

base = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw")

files = {
    "rule_of_law.csv": base / "rule_of_law.csv",
    "democracy_index.csv": base / "democracy_index.csv",
    "gdp_per_capita.csv": base / "gdp_per_capita.csv",
}

for name, path in files.items():
    print("\n" + "="*80)
    print(f"{name}  →  {path.resolve()}")
    df = pd.read_csv(path)
    print("Columns:", list(df.columns))
    # quick peeks
    print("Entities sample:", df["Entity"].dropna().unique()[:8])
    print("Year range:", int(df["Year"].min()), "→", int(df["Year"].max()))
