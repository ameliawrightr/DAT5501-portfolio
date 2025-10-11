import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.lines import Line2D

#helpers
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    col_entity = "Entity"
    col_year = "Year"
    rol_cols = [c for c in df.columns if "Rule of Law index" in c]
    if not rol_cols:
        raise ValueError("Could not find 'Rule of Law index' column")
    col_rol = rol_cols[0]
    df = df[[col_entity, col_year, col_rol]].dropna(subset=[col_entity, col_year, col_rol])
    df[col_year] = df[col_year].astype(int)
    df[col_rol] = df[col_rol].astype(float)
    return df, col_entity, col_year, col_rol

def set_matplotlib_defaults():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

#build a continuous Germany series by combining East/West when needed and interpolating gaps.
def germany_continuous(df, col_entity, col_year, col_rol, start=1930, end=1950):
    years = np.arange(start, end + 1)
    vals = []
    for y in years:
        subset = df[(df[col_year] == y) & (df[col_entity].str.contains("Germany"))]
        exact = subset[subset[col_entity] == "Germany"]
        if not exact.empty:
            vals.append(float(exact[col_rol].mean()))
        else:
            vals.append(float(subset[col_rol].mean()) if not subset.empty else np.nan)
    series = pd.DataFrame({col_year: years, col_rol: vals})
    series[col_rol] = series[col_rol].interpolate(method="linear", limit_direction="both")
    return series

def value_at(df_country, col_year, col_rol, year):
    yrs = df_country[col_year].values.astype(float)
    vals = df_country[col_rol].values.astype(float)
    return float(np.interp(year, yrs, vals))

#Δ vs t0 window for regime-start charts
def delta_since_start(df, col_entity, col_year, col_rol, country, start_year, horizon=12):
    s = df[df[col_entity] == country].sort_values(col_year)
    if s.empty:
        return pd.DataFrame({})
    base = value_at(s, col_year, col_rol, start_year)
    w = s[(s[col_year] >= start_year) & (s[col_year] <= start_year + horizon)].copy()
    w["t"] = w[col_year] - start_year
    w["delta_pts"] = w[col_rol] - base
    return w[["t", "delta_pts"]]


#figure generators

#FIGURE 1: Germany with event markers and shaded areas
def fig1_germany(df, col_entity, col_year, col_rol, outpath):
    ger_c = germany_continuous(df, col_entity, col_year, col_rol, 1930, 1950)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ger_c[col_year], ger_c[col_rol], linewidth=1.6, color="red")
    ax.set_title("Germany — Rule of Law (1930–1950)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rule of Law Index")

    #shadings
    ax.axvspan(1932, 1938, alpha=0.05, color="red", label="Rise of Nazi Germany")  # rise of Nazi Germany
    ax.axvspan(1939, 1945, alpha=0.20, color="grey", label="World War II")  # WWII

    #marker labels mid-height with slight x-offset to avoid dashed lines
    y_min, y_max = float(np.nanmin(ger_c[col_rol])), float(np.nanmax(ger_c[col_rol]))
    y_mid = y_min + 0.5*(y_max - y_min)
    for x, label in [(1932, "Nazi Germany rise (1932)"),
                     (1933, "Hitler becomes Chancellor (1933)"),
                     (1935, "Nuremberg Laws (1935)"),
                     (1939, "WWII starts (1939)"),
                     (1943, "Weakening of Nazi Germany begins (1943)"),
                     (1945, "WWII ends (1945)") ]:
        ax.axvline(x, linestyle="--", linewidth=0.8, color="grey")
        ax.text(x + 0.35, y_mid, label, rotation=90, va="center", ha="center", color="grey", fontsize=7.5)

    #label legend
    ax.legend(loc="upper right", fontsize=7.5)

    #every year tick
    ax.set_xlim(1930, 1950)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


#FIGURE 2: Russia with event markers
def fig2_russia(df, col_entity, col_year, col_rol, outpath):
    full_rus = df[df[col_entity] == "Russia"].sort_values(col_year)
    rus = full_rus[(full_rus[col_year].between(1999, 2024))]  # include 1999

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rus[col_year], rus[col_rol], linewidth=1.6, color="black")
    ax.set_title("Russia — Rule of Law (1999–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rule of Law Index")

    #shadings
    ax.axvspan(1999, 2000, alpha=0.03, color="red", label="Putin's Rise")  #Putin's Rise
    ax.axvspan(2000, 2003, alpha=0.10, color="red", label="Economic Reforms")  # Economic Reforms
    ax.axvspan(2019, 2021, alpha=0.03, color="purple", label="Constitutional Changes")  # Constitutional Changes
    ax.axvspan(2022, 2024, alpha=0.20, color="grey", label="Ukraine Invasion")  # Ukraine Invasion

    #clamp y-lims to data with small pad
    y_min = float(rus[col_rol].min())
    y_max = float(rus[col_rol].max())
    pad = 0.04 * (y_max - y_min if y_max > y_min else 0.01)
    ax.set_ylim(y_min - pad, y_max + pad)

    def y_at(year):
        return value_at(full_rus, col_year, col_rol, year)
    
    def place_label(ax, x, y_line, placed, y_low, y_high,
                    base_pos=0.6, 
                    min_gap_y=0.06,
                    min_gap_lbl=0.08,
                    search_steps=12):
        #find position for label that does not overlap with existing ones
        y0, y1 = ax.get_ylim()
        yr = y1 - y0

        y_cand = y0 + base_pos * yr
        y_cand = max(min(y_cand, y_high - 0.01*yr), y_low + 0.01*yr)
    
        #initial candiate y around base_pos of axis, clamp to y_low/y_high
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_base = ax.get_ylim()[0] + 0.55 * y_range

        #if too close to data line, move away
        if abs(y_cand - y_line) < min_gap_y * yr:
            if y_line <= y_cand:
                y_cand = min(y_high - 0.01*yr, y_line + min_gap_y * yr)
            else:
                y_cand = max(y_low + 0.01*yr, y_line - min_gap_y * yr)
        
        #if still too close to existing labels, try to move up/down in steps
        near = [(xx, yy) for xx, yy in placed if abs(xx - x) <= 1.0]
        step = (min_gap_lbl * yr)
        direction = 1
        for _ in range(search_steps):
            ok = True
            for _, yy in near:
                if abs(y_cand - yy) < min_gap_lbl * yr:
                    ok = False
                    break
                if ok:
                    return y_cand
                #not ok, move
                y_cand = y_cand + direction * step
                #flip direction if hit bounds
                direction *= -1
                step *= 0.9
                y_cand = max(min(y_cand, y_high - 0.01*yr), y_low + 0.01*yr)
        
        return y_cand  #give up, return last candidate
    
    #marker labels with collision avoidace
    y0, y1 = ax.get_ylim()
    yrange = y1 - y0
    y_low = y0 + 0.1 * yrange
    y_high = y1 - 0.06 * yrange
    
    markers = [
        (1999, "Putin to Prime Minister (1999)"),
        (2001.5, "Putin economic reforms (2001-2003)"),
        (2004, "Putin second term (2004)"),
        (2012, "Protests to Putin third term (2012)"),
        (2014, "Crimea (2014)"),
        (2018, "Putin fourth term (2018)"),
        (2020, "Constitutional changes (2020)"),
        (2022, "Full-scale invasion of Ukraine (2022)"),
        (2024, "Putin fifth term (2024)"),
    ]
    
    placed = []
    for i, (x, label) in enumerate(markers):
        ax.axvline(x, linestyle="--", linewidth=0.8, color="grey")

        #small x offset to avoid dashed line
        xoff = 0.35 if i % 2 == 0 else 0.45

        y_line = y_at(x)
        y_lab = place_label(ax, x, y_line, placed, y_low, y_high,
                            base_pos=0.58, min_gap_y=0.06, min_gap_lbl=0.08)
        placed.append((x, y_lab))


        ax.text(x + xoff, y_lab, label, 
                rotation=90, va="center", ha="center", 
                color="grey", fontsize=8, clip_on=True)

    #label legend
    ax.legend(loc="upper right", fontsize=7.5)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(1999, 2024)
    ax.grid(True, linewidth=0.4, alpha=0.25)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


#FIGURE 3: Δ vs t0 with top baseline
def fig3_since_regime_topbaseline(df, col_entity, col_year, col_rol, outpath):
    g_reg = delta_since_start(df, col_entity, col_year, col_rol, "Germany", 1933, horizon=12)
    r_reg = delta_since_start(df, col_entity, col_year, col_rol, "Russia", 1999, horizon=12)

    fig, ax = plt.subplots(figsize=(10, 6))
    if not g_reg.empty:
        ax.plot(g_reg["t"], g_reg["delta_pts"], linewidth=1.8, color="red", label="Germany (since 1933)")
    if not r_reg.empty:
        ax.plot(r_reg["t"], r_reg["delta_pts"], linewidth=1.8, color="black", label="Russia (since 1999)")

    # Determine negative extent only
    min_val = 0.0
    for s in [g_reg, r_reg]:
        if not s.empty:
            min_val = min(min_val, float(np.nanmin(s["delta_pts"])))
    extent = abs(min_val)
    pad = 0.08 * extent if extent > 0 else 0.1

    # Set 0 at the TOP (baseline) and only show negative values downward
    ax.set_ylim(0 - 1e-9, -(extent + pad))  # small epsilon to keep baseline visible

    # Style the TOP spine as baseline and add a top x-axis
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_linewidth(1.6)
    ax.spines["top"].set_color("grey")

    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xlabel("Baseline at t0", color="grey")
    ax_top.tick_params(axis='x', colors='grey')
    ax_top.spines['top'].set_color('grey')
    ax_top.spines['top'].set_linewidth(1.6)

    # Baseline handle for legend
    baseline_proxy = Line2D([0], [0], color="grey", linewidth=1.8)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(baseline_proxy)
    labels.append("Baseline (value at t0)")
    ax.legend(handles, labels, loc="best")

    ax.set_title("Change in Rule of Law — Years Since Regime Start (Δ index points)")
    ax.set_xlabel("Years since start (t)")
    ax.set_ylabel("Δ (index points; downward = deterioration)")

    # Integer ticks for t
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Endpoint labels
    def label_endpoint(s, color):
        if s.empty: return
        t_end = s["t"].max()
        v_end = float(s.loc[s["t"] == t_end, "delta_pts"].iloc[0])
        ax.text(t_end, v_end, f"{v_end:+.2f}", va="center", ha="left", fontsize=8, color=color)

    label_endpoint(g_reg, "red")
    label_endpoint(r_reg, "black")

    # Light y-grid only
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.2)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def fig4_grouped(df, col_entity, col_year, col_rol, outpath):
    # Period windows
    g_pre = df[(df[col_entity].str.contains("Germany")) & (df[col_year].between(1930, 1932))][col_rol].mean()
    g_war = df[(df[col_entity].str.contains("Germany")) & (df[col_year].between(1939, 1945))][col_rol].mean()
    r_pre = df[(df[col_entity] == "Russia") & (df[col_year].between(2010, 2018))][col_rol].mean()
    r_war = df[(df[col_entity] == "Russia") & (df[col_year].between(2022, 2024))][col_rol].mean()

    x = np.arange(2)
    width = 0.42

    fig, ax = plt.subplots(figsize=(10, 6))
    # Four explicit legend entries
    b_g_pre = ax.bar(x[0] - width/2, g_pre, width, color="red", alpha=0.5, label="Pre-war Germany")
    b_g_war = ax.bar(x[0] + width/2, g_war, width, color="red", alpha=1.0, label="War Germany")
    b_r_pre = ax.bar(x[1] - width/2, r_pre, width, color="black", alpha=0.5, label="Pre-war Russia")
    b_r_war = ax.bar(x[1] + width/2, r_war, width, color="black", alpha=1.0, label="War Russia")

    ax.set_title("Rule of Law — Pre vs War Period Averages (Grouped)")
    ax.set_ylabel("Rule of Law Index")
    ax.set_xticks(x)
    ax.set_xticklabels(["Germany", "Russia"])
    ax.legend(loc="best", title="Series")

    # No grid (no dashed feel)
    ax.grid(False)

    # Shared vertical centerline for Δ labels
    vals = [g_pre, g_war, r_pre, r_war]
    y_min = min(vals)
    y_max = max(vals)
    y_centerline = y_min + 0.5*(y_max - y_min)

    def center_delta_text(b_pre, b_war, label_country):
        y0 = b_pre.patches[0].get_height()
        y1 = b_war.patches[0].get_height()
        x_mid = (b_pre.patches[0].get_x() + b_pre.patches[0].get_width()/2 +
                 b_war.patches[0].get_x() + b_war.patches[0].get_width()/2) / 2
        ax.text(x_mid, y_centerline, f"Δ {label_country}: {y1 - y0:+.2f}",
                ha="center", va="center", fontsize=9)

    center_delta_text(b_g_pre, b_g_war, "Germany")
    center_delta_text(b_r_pre, b_r_war, "Russia")

    # Bar value labels
    for cont in [b_g_pre, b_g_war, b_r_pre, b_r_war]:
        v = cont.patches[0].get_height()
        ax.text(cont.patches[0].get_x() + cont.patches[0].get_width()/2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate rule-of-law figures (Germany & Russia).")
    ap.add_argument("--csv", default="/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/data/raw/rule_of_law.csv", help="Path to rule_of_law.csv")
    ap.add_argument("--outdir", default="/Users/amelia/DAT5501-portfolio/lab06_rule_of_law_group_project/artifacts/figures", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_matplotlib_defaults()
    df, col_entity, col_year, col_rol = load_data(args.csv)

    fig1_germany(df, col_entity, col_year, col_rol, outdir / "fig1_germany_1930_1950.png")
    fig2_russia(df, col_entity, col_year, col_rol, outdir / "fig2_russia_1999_2024.png")
    fig3_since_regime_topbaseline(df, col_entity, col_year, col_rol, outdir / "fig3_since_regime_start_topbaseline.png")
    fig4_grouped(df, col_entity, col_year, col_rol, outdir / "fig4_grouped_pre_vs_war.png")

    print("Saved:")
    print(outdir / "fig1_germany_1930_1950.png")
    print(outdir / "fig2_russia_1999_2024.png")
    print(outdir / "fig3_since_regime_start_topbaseline.png")
    print(outdir / "fig4_grouped_pre_vs_war.png")

if __name__ == "__main__":
    main()
