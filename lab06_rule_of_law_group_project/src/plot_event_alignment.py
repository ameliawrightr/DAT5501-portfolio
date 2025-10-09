from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

raw = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw")
out = Path("/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/artifacts"); out.mkdir(parents=True,exist_ok=True)

dem = pd.read_csv(raw / "democracy_index.csv")
rol = pd.read_csv(raw / "rule_of_law.csv")

#identify correct columns
dem_col = [c for c in dem.columns if "liberal democracy" in c.lower()][0]
rol_col = [c for c in rol.columns if "rule of law" in c.lower()][0]

#ensure numeric
dem[dem_col] = pd.to_numeric(dem[dem_col], errors="coerce")
rol[rol_col] = pd.to_numeric(rol[rol_col], errors="coerce")

#Germany 1933–1946
g = dem[(dem["Entity"]=="Germany") & dem["Year"].between(1933, 1946)][["Year", dem_col]].copy()

#Russia 2014–2024
r = rol[(rol["Entity"]=="Russia") & rol["Year"].between(2014, 2024)][["Year", rol_col]].copy()

print("Germany years:", g["Year"].min(), "→", g["Year"].max())
print("Russia years:", r["Year"].min(), "→", r["Year"].max())

#calculate % change relative to baseline to create a synthetic "t" axis measured in years from war start
#Germany baseline 1936–1938
g_base = g[g["Year"].between(1936, 1938)][dem_col].mean()
g["pct_change"] = (g[dem_col] - g_base) / abs(g_base) * 100
g["t"] = g["Year"] - 1939   # war onset = 1939

#Russia baseline 2019–2021
r_base = r[r["Year"].between(2019, 2021)][rol_col].mean()
r["pct_change"] = (r[rol_col] - r_base) / abs(r_base) * 100
r["t"] = r["Year"] - 2022   # war onset = 2022

#plot aligned trajectories
plt.figure(figsize=(9,5))
plt.plot(g["t"], g["pct_change"], label="Germany (WWII)", linewidth=2, color="black")
plt.plot(r["t"], r["pct_change"], label="Russia (Ukraine war)", linewidth=2, color="red")

plt.axvline(0, color="k", ls="--", alpha=0.6)
plt.text(0.1, plt.ylim()[1]*0.9, "War begins (t=0)", fontsize=8, rotation=90)

plt.title("Rule of Law Erosion Aligned to War Onset")
plt.xlabel("Years since war start (t = 0)")
plt.ylabel("Change from pre-war baseline (%)")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out / "event_time_alignment.png", dpi=300)
plt.close()
print("Saved: event_time_alignment.png")

