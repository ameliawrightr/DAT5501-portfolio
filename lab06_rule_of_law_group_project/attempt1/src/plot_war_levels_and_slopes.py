from importlib import import_module
from importlib.machinery import SourceFileLoader
from types import ModuleType
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

HERE = Path(__file__).resolve().parent
#MOD_PATH = HERE / "canonical_country_build.py"
MOD_PATH = "/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/src/roltools/canonical_country_build.py"

DATA = HERE / "data"
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


try:
    import canonical_country_build as canon
except ModuleNotFoundError:
    if not MOD_PATH.exists():
        raise FileNotFoundError(f"Couldn't find canonical_country_build.py at {MOD_PATH}")
    loader = SourceFileLoader("canonical_country_build", str(MOD_PATH))
    canon = ModuleType(loader.name)
    loader.exec_module(canon)

german_canonical = canon.german_canonical
russia_canonical = canon.russia_canonical

rol = pd.read_csv((Path(__file__).resolve().parents[1] / "data" / "rule_of_law_index.csv"))


def load_canon():
    try:
        return import_module("canonical_country_build")
    except Exception:
        pass
    for p in MOD_PATH.glob("**/canonical_country_build.py"):
        if p.exists():
            mod = ModuleType("canonical_country_build")
            loader = SourceFileLoader("canonical_country_build", str(p))
            loader.exec_module(mod)
            return mod
    if str(HERE) not in sys.path:
        sys.path.insert(0,str(HERE))
    return import_module("canonical_country_build")

  

def detect_rol_col(cols):
    for c in cols:
        if c.strip().lower() == "rule of law index (central estimate)":
            return c
    # otherwise any column containing "rule of law"
    for c in cols:
        if "rule of law" in c.lower():
            return c
    raise KeyError(f"Couldn't find a Rule of Law column. Available: {list(cols)}")



rol_col = detect_rol_col(rol.columns)

#detect cols + make numeric
rol["Year"] = pd.to_numeric(rol["Year"], errors="coerce")
rol[rol_col] = pd.to_numeric(rol[rol_col], errors="coerce")

canon = load_canon()
german_canonical = canon.german_canonical
russia_canonical = canon.russia_canonical

g_df = german_canonical(rol, rol_col)
r_df = russia_canonical(rol, rol_col)

print("DEBUG: Germany canonical 1949-1953")
print(g_df[g_df["Year"].between(1949, 1953)].to_string(index=False))


#Calculate means and slopes for each country
def mean_in_range(df, col, start, end):
    #Calculate mean of a column in a given year range.
    m = df[df["Year"].between(start, end)].dropna(subset=[col])
    if m.empty:
        print(f"WARNING: No data in range {start}-{end}.")
        return float("nan"), (start, end), 0
    return float(m[col].mean()), (int(m["Year"].min()), int(m["Year"].max())), len(m)

def slope_in_range(df, col, start, end):
    #Simple OLS slope (per year) of col vs Year over [start, end]
    #Return Nan if < 2 data points
    m = df[df["Year"].between(start, end)].dropna(subset=[col,"Year"])
    if len(m) < 2:
        print(f"WARNING: Not enough data points to calculate slope in range {start}-{end}.")
        return float("nan"), (start, end), 0
    x, y = m["Year"].to_numpy(), m[col].to_numpy()
    #polyfit order 1 => slope
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, (int(m["Year"].min()), int(m["Year"].max())), len(m)

def annotate_mean(ax, xmid, y, label):
    if is_finite(xmid, y):
        ax.scatter([xmid], [y], s=30)
        ax.text(xmid, y, f" {label}", va="center")

def shade(ax, a, b, color, alpha=0.25):
    if np.isfinite(a) and np.isfinite(b):
        ax.axvspan(a, b, color=color, alpha=alpha, zorder=0)

def pct_change(new, old):
    #returns Nan if any input is Nan or old==0
    if not (np.isfinite(new) and np.isfinite(old)) or old == 0:
        return float("nan")
    return (new - old) / old * 100

def is_finite(*vals):
    return all(np.isfinite(v) for v in vals)

def label_band_mean(ax, df, col, a, b, tag, y_off=0.015):
    m, used, n = mean_in_range(df, col, a, b)
    if np.isfinite(m) and np.isfinite(used[0]) and np.isfinite(used[1]):
        xm = (used[0] + used[1]) / 2
        ax.scatter([xm], [m], s=18, zorder=3)
        ax.text(xm, m + y_off, f"{tag}: {m:.2f} (n={n})", va="bottom", ha="center", fontsize=9)
    return m, used

#diagnostic 
def non_nan_span(df, col, name):
    d = df.dropna(subset=[col])
    return f"{name}: {int(d['Year'].min())}-{int(d['Year'].max())} ({len(d)})" if not d.empty else f"{name}: no non-NaN"

print("Available (non-NaN) spans:")
print(" ", non_nan_span(g_df, rol_col, "Germany (canonical)"))
print(" ", non_nan_span(r_df, rol_col, "Russia (canonical)"))

def span_str(used):
    a, b = used
    return f"{int(a)}-{int(b)}" if np.isfinite(a) and np.isfinite(b) else "-"

#periods 
g_pre  = (1928, 1932)
g_war  = (1939, 1945)
g_post = (1949, 1953)

r_pre  = (2010, 2013)
r_war  = (2014, 2021)
r_post = (2022, 2024) 

#Fig A: dual-panel line plots of rule of law over time, with means annotated
C_PRE, C_WAR, C_POST = "lightblue", "lightcoral", "lightgreen"

fig, axes = plt.subplots(1,2, figsize=(11,4), sharey=True)

#GERMANY - use rule of law index
ax = axes[0]
ax.plot(g_df["Year"], g_df[rol_col], linewidth=1.8, color="black")
shade(ax, *g_pre, C_PRE)
shade(ax, *g_war, C_WAR)
shade(ax, *g_post, C_POST)
gm_pre, _ = label_band_mean(ax, g_df, rol_col, *g_pre, "pre")
gm_war, _ = label_band_mean(ax, g_df, rol_col, *g_war, "war")
gm_post, _ = label_band_mean(ax, g_df, rol_col, *g_post, "post")
g_drop = pct_change(gm_war, gm_pre)
if np.isfinite(g_drop):
    ax.text(g_war[0], ax.get_ylim()[1] - 0.02, f"Δ level (war vs pre): {g_drop:.1f}%", fontsize=9)
ax.set_title("Germany - WWII mobilisation -> institutional collapse")
ax.set_xlim(1920, 1960)
ax.set_ylabel("Rule of Law Index")
ax.set_xlabel("Year")

#RUSSIA - use rule of law index
ax = axes[1]
ax.plot(r_df["Year"], r_df[rol_col], linewidth=1.8, color="black")
shade(ax, *r_pre, C_PRE)
shade(ax, *r_war, C_WAR)
shade(ax, *r_post, C_POST)
rm_pre, _ = label_band_mean(ax, r_df, rol_col, *r_pre, "pre")
rm_war, _ = label_band_mean(ax, r_df, rol_col, *r_war, "war")
rm_post, _ = label_band_mean(ax, r_df, rol_col, *r_post, "post")
r_drop = pct_change(rm_war, rm_pre)
if np.isfinite(r_drop):
    ax.text(r_war[0], ax.get_ylim()[1] - 0.02, f"Δ level (war vs pre): {r_drop:.1f}%", fontsize=9)
ax.set_title("Russia - Ukraine war -> institutional decline")
ax.set_xlim(2000, 2025)

fig.suptitle("War Mobilisation Erodes Rule of Law (Germany vs Russia)", fontsize=12)
plt.tight_layout()
plt.savefig(ART / "war_levels_dual_panel.png", dpi=300, bbox_inches="tight")
plt.close()

#Fig B: slopes (pre vs war)
gs_pre, _, _ = slope_in_range(g_df, rol_col, *g_pre)
gs_war, _, _ = slope_in_range(g_df, rol_col, *g_war)
rs_pre, _, _ = slope_in_range(r_df, rol_col, *r_pre)
rs_war, _, _ = slope_in_range(r_df, rol_col, *r_war)

labels = ["Germany pre", "Germany war", "Russia pre", "Russia war"]
slopes  = [gs_pre, gs_war, rs_pre, rs_war]
colors  = [C_PRE, C_WAR, C_PRE, C_WAR]

fig, ax = plt.subplots(figsize=(7.5, 3.4))
bars = ax.bar(labels, slopes, color=colors)
ax.axhline(0, lw=1, color="#999")
for b in bars:
    y = b.get_height()
    if np.isfinite(y):
        ax.text(b.get_x()+b.get_width()/2, y, f"{y:.3f}/yr", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Slope (index points per year)")
ax.set_title("War Accelerates Rule of Law Decline")
plt.tight_layout()
plt.savefig(ART / "slopes.png", dpi=300, bbox_inches="tight")
plt.close()


#Fig C: normalised overlay (align at mobilisation)
def normalise_and_align(df, col, pre_window, t0):
    pre_mean, _, _ = mean_in_range(df, col, *pre_window)
    out = df.dropna(subset=[col]).copy()
    out["norm"] = out[col] / pre_mean if np.isfinite(pre_mean) else np.nan
    out["t"] = out["Year"] - t0
    return out[["t", "norm"]].dropna()

g_norm = normalise_and_align(g_df, rol_col, g_pre, g_war[0])
r_norm = normalise_and_align(r_df, rol_col, r_pre, r_war[0])

fig, ax = plt.subplots(figsize=(7.5,4))
ax.plot(g_norm["t"], g_norm["norm"], lw=2, label="Germany", color="black")
ax.plot(r_norm["t"], r_norm["norm"], lw=2, label="Russia", color="red")
ax.axvline(0, color="#444", lw=1, ls="--")
ax.axhline(1, color="#999", lw=1, ls="--")
ax.set_xlim(-10, 15)
ax.set_ylim(0, None)
ax.set_xlabel("Years since mobilisation")
ax.set_ylabel("Rule of Law Index (normalised to pre-war mean)")
ax.set_title("Rule of Law Decline After War Mobilisation (Normalised)")
ax.legend()
plt.tight_layout()
plt.savefig(ART / "war_levels_normalised.png", dpi=300, bbox_inches="tight")
plt.close()