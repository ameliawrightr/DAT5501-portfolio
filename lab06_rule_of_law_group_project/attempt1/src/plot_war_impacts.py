from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

raw = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw")
out = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/artifacts"); out.mkdir(parents=True, exist_ok=True)

rol_col = "Rule of Law Index (central estimate)"

from roltools import germany_canonical, russia_canonical


def mean_in_range(df, col, start, end):
    m = df[df["Year"].between(start, end)].dropna(subset=[col])
    if m.empty:
        print(f"WARNING: No data in range {start}-{end}.")
        return float("nan"), (start, end), 0
    return float(m[col].mean()), (int(m["Year"].min()), int(m["Year"].max())), len(m)

def slope_in_range(df, col, start, end):
    m = df[df["Year"].between(start, end)].dropna(subset=[col, "Year"])
    if len(m) < 2:
        print(f"WARNING: Not enough points for slope in {start}-{end}.")
        return float("nan"), (start, end), 0
    x, y = m["Year"].to_numpy(), m[col].to_numpy()
    s = float(np.polyfit(x, y, 1)[0])
    return s, (int(m["Year"].min()), int(m["Year"].max())), len(m)

def pct_change(new, old):
    if not (np.isfinite(new) and np.isfinite(old)) or old == 0:
        return float("nan")
    return (new / old - 1.0) * 100.0

def is_finite(*vals): 
    return all(np.isfinite(v) for v in vals)


#load rule of law index 
rol = pd.read_csv(raw / "rule_of_law.csv")

g_all = germany_canonical(rol, rol_col)
r_all = russia_canonical(rol, rol_col)

rol[rol_col] = pd.to_numeric(rol[rol_col], errors="coerce")

#filter Germany for 1918-1950 period,
#main Germany ≤1945 and West Germany 1949-1950
g_slice = g_all[g_all["Year"].between(1949, 1953)][["Year", rol_col]].dropna().copy()
if g_slice.empty:
    raise RuntimeError("Germany canonical slice is empty; check canonical builder and rol_col.")

print("Final WWII series years:", g_slice["Year"].min(), "→", g_slice["Year"].max(), "rows:", len(g_slice))

#annotation of key historical events
events = [
    (1919, "Weimar Constitution"),
    (1933, "Nazi rise to power"),
    (1935, "Nuremberg Laws"),
    (1939, "WWII begins"),
    (1945, "WWII ends / Allied occupation"),
    (1949, "FRG/GDR formed")
]

#plot german
plt.figure(figsize=(10,5.5))
plt.plot(g_slice["Year"], g_slice[rol_col], linewidth=2, color="black")

y_top = float(g_slice[rol_col].max())
for yr, label in events:
    if g_slice["Year"].min() <= yr <= g_slice["Year"].max():
        plt.axvline(yr, ls="--", alpha=0.35)
        plt.text(yr+0.15, y_top, label, rotation=90, va="top", fontsize=8)

plt.title("Germany: Collapse of Rule of Law (1918-1950)")
plt.xlabel("Year")
plt.ylabel("Rule of Law Index")
plt.ylim(0,1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out / "germany_WW2_dual.png", dpi=300)
plt.close()

print("Saved: germany_WW2_dual.png")


#Russia 2010-2024
russia = rol[(rol["Entity"] == "Russia") & (rol["Year"].between(2010, 2024))][["Year", rol_col]].dropna()
russia = russia.sort_values("Year")

plt.figure(figsize=(10,5.5))
plt.plot(russia["Year"], russia[rol_col], linewidth=2,color="red")

#annotations for key political events
for yr, label in [
    (2012, "Protests / centralisation"),
    (2014, "Crimea annexation"),
    (2020, "Constitutional term reset"),
    (2022, "Ukraine invasion"),
]:
    if russia["Year"].min() <= yr <= russia["Year"].max():
        plt.axvline(yr, ls="--", alpha=0.4)
        y_top = russia[rol_col].max()
        plt.text(yr+0.1, y_top, label, rotation=90, va="top",fontsize=8)

plt.title("Russia: Erosion of Rule of Law (2010-2024)")
plt.xlabel("Year")
plt.ylabel("Rule of Law Index")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out / "russia_current_rol.png", dpi=300)
plt.close()

print("Saved: russia_current_rol.png")