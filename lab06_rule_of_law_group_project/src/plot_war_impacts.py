from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

raw = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw")
out = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/artifacts"); out.mkdir(parents=True, exist_ok=True)

#load democracy index 
dem = pd.read_csv(raw / "democracy_index.csv")
#load rule of law index 
rol = pd.read_csv(raw / "rule_of_law.csv")

#identify correct cols
dem_col = [c for c in dem.columns if "liberal democracy" in c.lower()][0]
rol_col = [c for c in rol.columns if "rule of law" in c.lower()][0]
print("Democracy column:", dem_col)
print("Rule of law column:", rol_col)

dem[dem_col] = pd.to_numeric(dem[dem_col], errors="coerce")
rol[rol_col] = pd.to_numeric(rol[rol_col], errors="coerce")

#filter Germany for 1918-1950 period,
#main Germany ≤1945 and West Germany 1949-1950
germany_main = dem[(dem["Entity"] == "Germany") & dem["Year"].between(1918,1945)][["Year",dem_col]].copy()
west_49_50  = dem[(dem["Entity"]=="West Germany") & dem["Year"].between(1949,1950)][["Year",dem_col]].copy()
g_series = pd.concat([germany_main, west_49_50], ignore_index=True).sort_values("Year")

print("Final WWII series years:", g_series["Year"].min(), "→", g_series["Year"].max(), "rows:", len(g_series))

#annotation of key historical events
events = [
    (1919, "Weimar Constitution"),
    (1933, "Nazi rise to power"),
    (1935, "Nuremberg Laws"),
    (1939, "WWII begins"),
    (1945, "WWII ends / Allied occupation"),
    (1949, "FRG / GDR formed")
]

#plot german
plt.figsize=(10,5.5)
plt.plot(g_series["Year"], g_series[dem_col], linewidth=2, color="black")

y_top = float(g_series[dem_col].max())
for yr, label in events:
    if g_series["Year"].min() <= yr <= g_series["Year"].max():
        plt.axvline(yr, ls="--", alpha=0.35)
        plt.text(yr+0.15, y_top, label, rotation=90, va="top", fontsize=8)

plt.title("Germany: Collapse of Rule of Law (1918-1950) via Democracy Proxy")
plt.xlabel("Year")
plt.ylabel("Liberal Democracy Index")
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