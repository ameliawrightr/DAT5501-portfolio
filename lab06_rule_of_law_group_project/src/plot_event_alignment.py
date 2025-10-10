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

#Germany (late Weimar -> WW2 -> occupation)
#calculate % change relative to baseline to create a synthetic "t" axis measured in years from war start
g = dem[(dem["Entity"]=="Germany") & dem["Year"].between(1928, 1946)][["Year", dem_col]].copy()
g = g.dropna(subset=[dem_col]).sort_values("Year")
if g.empty:
    raise ValueError("Germany slice is empty; check democracy dataset entity/years.")

print("Germany years:", g["Year"].min(), "→", g["Year"].max())

#baseline = late Weimar, pre authoritarian 
g_base = g[g["Year"].between(1928, 1932)][dem_col].mean()
if pd.isna(g_base) or g_base < 0.05:
    raise ValueError(f"Germany baseline invalid: {g_base}")
g["pct_change"] = (g[dem_col] / g_base - 1) * 100
g["t"] = g["Year"] - 1939   #t=0 at war start (1939)
g["pct_change"] = g["pct_change"].rolling(3, center=True, min_periods=1).mean() #smooth series for readability

#Russia 2014–2024 (pre-war -> invasion -> now)
r = rol[(rol["Entity"]=="Russia") & rol["Year"].between(2014, 2024)][["Year", rol_col]].copy()
r = r.dropna(subset=[rol_col]).sort_values("Year")
if r.empty:
    raise ValueError("Russia slice is empty; check rule_of_law.csv dataset entity/years.")

print("Russia years:", r["Year"].min(), "→", r["Year"].max())

#baseline 2019–2021
r_base = r[r["Year"].between(2019,2021)][rol_col].mean()
if pd.isna(r_base) or r_base == 0:
    raise ValueError(f"Russia basline invalid: {r_base}")
r["pct_change"] = (r[rol_col] / r_base - 1) * 100
r["t"] = r["Year"] - 2022   # war onset = 2022
r["pct_change"] = r["pct_change"].rolling(3, center=True, min_periods=1).mean()


#plot 
t_min, t_max = -6,6
g_plot = g[(g["t"] >= t_min) & (g["t"] <= t_max)]
r_plot = r[(r["t"] >= t_min) & (r["t"] <= t_max)]

plt.figure(figsize=(10,5.5))
plt.plot(g_plot["t"], g_plot["pct_change"], label="Germany (WWII)", linewidth=2, color="black")
plt.plot(r_plot["t"], r_plot["pct_change"], label="Russia (Ukraine war)", linewidth=2, color="red")

plt.axvline(0, color="k", ls="--", alpha=0.6)
plt.axhline(0, color="grey", ls=":", alpha=0.6)

plt.title("Rule of Law Erosion Aligned to War Onset")
plt.xlabel("Years since war start (t = 0)")
plt.ylabel("Change from pre-war baseline (%)")

plt.xlim(-6,6)
plt.ylim(min(g["pct_change"].min(),r["pct_change"].min())*1.1,10)
plt.annotate("War begins", xy=(0,0), xytext=(0.3, plt.ylim()[1]*0.8),
             arrowprops=dict(arrowstyle="->",alpha=0.6),fontsize=9)


plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out / "event_time_alignment.png", dpi=300)
plt.close()
print("Saved: event_time_alignment.png")

#SMALL DIAGNOSITICS
g_t0 = g.loc[g["t"]==0, "pct_change"]
r_t0 = r.loc[r["t"]==0, "pct_change"]
print("Saved:", out / "event_time_alignment.png")
if not g_t0.empty and not r_t0.empty:
    print("Germany t=0 value (%):", g_t0.iloc[0])
    print("Russia t=0 value (%):", r_t0.iloc[0])