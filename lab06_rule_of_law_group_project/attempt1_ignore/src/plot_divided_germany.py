from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

raw = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw")
out = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/artifacts"); out.mkdir(parents=True, exist_ok=True)

rol = pd.read_csv(raw / "rule_of_law.csv")

rol_col = "Rule of Law Index (central estimate)"

from roltools import germany_canonical, russia_canonical

#load rol file
df = pd.read_csv(raw / "rule_of_law.csv")

#detect value col robustly
val = rol_col
ew = rol[rol["Entity"].isin(["East Germany", "West Germany"])][["Entity", "Year", val]].dropna()

#keep only east/west and div window
mask_eu = df["Entity"].isin(["East Germany", "West Germany"])
mask_years = df["Year"].between(1949,1990) #div period
sub = df[mask_eu & mask_years].dropna(subset=[val]).copy()

#plot two lines, one per entity
plt.figure(figsize=(10,5.5))
for name, g in sub.groupby("Entity"):
    g = g.sort_values("Year")
    plt.plot(g["Year"], g[val], linewidth=2, label=name)

#historical annotations 
for yr, label in [(1961, "Berlin Wall built"),
                  (1989, "Berlin Wall falls"),
                  (1990, "Reunification of Germany")]:
    plt.axvline(yr, ls="--", alpha=0.35)
    #place label near top
    y_top = sub[val].max()
    plt.text(yr+0.2, y_top, label, rotation=90, va="top", fontsize=8)

plt.title("Divided Germany (1949-1990): Rule of Law - West vs East")
plt.xlabel("Year"); plt.ylabel("Rule of Law index (higher = stronger)")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out / "divided_germany_east_vs_west.png", dpi=300)
plt.close

print("Saved:", out / "divided_germany_east_vs_west.png")