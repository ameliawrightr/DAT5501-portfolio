import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
        txt_colour = "black" if x in (1933, 1945) else "grey" #highlight key events
        ax.text(x + 0.35, y_mid, label, rotation=90, va="center", ha="center", color=txt_colour, fontsize=7.5)

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
    
    def delta_since_start(country, start_year, horizon=12):
        s = df[df[col_entity] == country].sort_values(col_year)
        if s.empty:
            return None
        yrs = s[col_year].values.astype(float)
        vals = s[col_rol].values.astype(float)
        base = float(np.interp(start_year, yrs, vals))
        w = s[(s[col_year] >= start_year) & (s[col_year] <= start_year + horizon)].copy()
        w["t"] = w[col_year] - start_year
        w["delta_pts"] = w[col_rol] - base
        return w[["t", "delta_pts"]]
    
    g_reg = delta_since_start("Germany", 1933, horizon=12)
    r_reg = delta_since_start("Russia", 1999, horizon=12)

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx() #right y-axis shares x but has own y-axis

    #plot series on own axes
    if g_reg is not None and not g_reg.empty:
        axL.plot(g_reg["t"], g_reg["delta_pts"], color="red", linewidth=1.8, label="Germany (since 1933)")
    if r_reg is not None and not r_reg.empty:
        axR.plot(r_reg["t"], r_reg["delta_pts"], color="black", linewidth=1.8, label="Russia (since 1999)")

    #determine negative extents (only show deterioration downward)
    def neg_extent(s):
        return abs(min(0.0, float(np.nanmin(s["delta_pts"])))) if s is not None and not s.empty else 0.0
    
    g_ext = neg_extent(g_reg)
    r_ext = neg_extent(r_reg)
    pad_g = 0.08 * g_ext if g_ext > 0 else 0.1
    pad_r = 0.08 * r_ext if r_ext > 0 else 0.1

    #put baseline at top on both axis, only negative downward
    axL.set_ylim(0 - 1e-9, -(g_ext + pad_g))  # small epsilon to keep baseline visible
    axR.set_ylim(0 - 1e-9, -(r_ext + pad_r))

    #style top baseline spine and legend entry
    for ax in (axL, axR):
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(1.6)
        ax.spines["top"].set_color("grey")

    baseline_proxy = Line2D([0], [0], color="grey", linewidth=1.8)

    #legend
    axL.set_title("Change in Rule of Law — Years Since Regime Start")
    axL.set_xlabel("Years since start (t)")
    axL.set_ylabel("Δ (Germany index points)", color="red")
    axR.set_ylabel("Δ (Russia index points)", color="black")

    #integer ticks on x
    axL.xaxis.set_major_locator(MaxNLocator(integer=True))

    #endpoint labels
    def label_endpoint(ax, s, color):
        if s is None or s.empty: 
            return
        t_end = s["t"].max()
        v_end = float(s.loc[s["t"] == t_end, "delta_pts"].iloc[0])
        ax.text(t_end, v_end, f"{v_end:+.2f}", va="center", ha="left", fontsize=8, color=color)

    label_endpoint(axL, g_reg, "red")
    label_endpoint(axR, r_reg, "black")

    #combined legend
    handlesL, labelsL = axL.get_legend_handles_labels()
    handlesR, labelsR = axR.get_legend_handles_labels()
    handles = handlesL + handlesR + [baseline_proxy]
    labels = labelsL + labelsR + ["Baseline (t=0)"]
    axL.legend(handles, labels, loc="best", fontsize=7.5)

    #light y-grid only
    axL.grid(True, axis="y", linewidth=0.4, alpha=0.2)
    axR.grid(True, axis="y", linewidth=0.4, alpha=0.2)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    

#FIGURE 3 (ALT): dual y-axes with % change from t0 (0 at top, only negatives)
#shows data in a easier to interpret % change format, but dual axes can be hard to read

def _pct_since_start(df, col_entity, col_year, col_rol, country, start_year, horizon=12):
    s = df[df[col_entity] == country].sort_values(col_year)
    if s.empty:
        return None, start_year
    
    yrs = s[col_year].values.astype(float)
    vals = s[col_rol].values.astype(float)
    base = float(np.interp(start_year, yrs, vals))

    w = s[(s[col_year] >= start_year) & (s[col_year] <= start_year + horizon)].copy()
    w["t"] = w[col_year] - start_year
    # % change vs t0
    w["delta_pct"] = (w[col_rol] / base - 1.0) * 100.0
    return w[["t", "delta_pct"]], start_year


def fig3a_since_regime_dual_axes_pct(df, col_entity, col_year, col_rol, outpath, g_start=1933, r_start=1999):

    g, _ = _pct_since_start(df, col_entity, col_year, col_rol, "Germany", g_start, horizon=12)
    r, _ = _pct_since_start(df, col_entity, col_year, col_rol, "Russia", r_start, horizon=12)

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()  #right axis for Russia

    if g is not None and not g.empty:
        axL.plot(g["t"], g["delta_pct"], color="red", linewidth=1.8, label="Germany (since 1933)")
    if r is not None and not r.empty:
        axR.plot(r["t"], r["delta_pct"], color="black", linewidth=1.8, label="Russia (since 1999)")

    # negative extents only (decline), put 0 at TOP on both axes
    def neg_extent(df_):
        return abs(min(0.0, float(np.nanmin(df_["delta_pct"])))) if (df_ is not None and not df_.empty) else 0.0
    
    def extents(df_):
            if df_ is None or df_.empty:
                return 0.0, 0.0
            v = df_["delta_pct"].to_numpy()
            v = v[~np.isnan(v)]
            if v.size == 0:
                return 0.0, 0.0
            pos = max(0.0, float(np.nanmax(v)))
            neg = min(0.0, float(np.nanmin(v)))
            return neg, pos

    g_neg, g_pos = extents(g)
    r_neg, r_pos = extents(r)  
    
    g_ext = neg_extent(g); r_ext = neg_extent(r)
    
    pad_g = 0.08 * g_ext if g_ext > 0 else 0.1
    pad_r = 0.08 * r_ext if r_ext > 0 else 0.1
    
    axL.set_ylim(g_pos + pad_g, g_neg - pad_g)
    axR.set_ylim(r_pos + pad_r, r_neg - pad_r)
    
    #% formatters, labels
    pct_fmt = FuncFormatter(lambda v, pos: f"{v:.0f}%")
    axL.yaxis.set_major_formatter(pct_fmt)
    axR.yaxis.set_major_formatter(pct_fmt)
    axL.set_title("% Change in Rule of Law — Years Since Regime Start")
    axL.set_xlabel("Years since start (t)")
    axL.set_ylabel("Δ % — Germany")
    axR.set_ylabel("Δ % — Russia")
    axL.xaxis.set_major_locator(MaxNLocator(integer=True))

    def y_label_pos(ax, frac=0.2):
        y0, y1 = ax.get_ylim()
        return y0 + frac * (y1 - y0)
    
     

    
    #event markers (years since start)
    g_marker_year = [(1934, "WWII starts")]
    r_marker_year = [(2003, "Russia economic reforms end")]
    
    #make absolute
    g_t = [(x - g_start, label) for x, label in g_marker_year]
    r_t = [(x - r_start, label) for x, label in r_marker_year]

    # Germany markers
    for x, label in g_t:
        axL.axvline(x, 0, 1, linestyle="--", linewidth=0.8, color="red", zorder=0)
        axL.text(x + 0.35, y_label_pos(axL), label,
                 rotation=90, va="center", ha="center", color="red", fontsize=8.5, clip_on=False)
        
    # Russia marker
    for x, label in r_t:
        axR.axvline(x, 0, 1, linestyle="--", linewidth=0.8, color="black", zorder=0)
        axR.text(x + 0.35, y_label_pos(axR), label,
                 rotation=90, va="center", ha="center", color="grey", fontsize=8.5, clip_on=False)

    # endpoint annotations
    def endpoint(ax, s, color):
        if s is None or s.empty:
            return
        t_end = s["t"].max()
        v_end = float(s.loc[s["t"] == t_end, "delta_pct"].iloc[0])
        ax.text(t_end, v_end, f"{v_end:+.1f}%", va="center", ha="left", fontsize=8, color=color)

    endpoint(axL, g, "red")
    endpoint(axR, r, "black")

    #every year tick
    axL.set_xlim(0, 12)
    axL.xaxis.set_major_locator(MultipleLocator(1))

    # combined legend
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="best", fontsize=7.5) 

    axL.grid(True, axis="y", linewidth=0.4, alpha=0.2)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


#FIGURE 4: Grouped bar chart pre vs war
#FIGURE 4: 
def fig4a_dual_axis(df, col_entity, col_year, col_rol, outpath):
    """
    Grouped bar chart (Pre vs War) with TWO Y-AXES
      - Left Y-axis: Germany (pre/war)
      - Right Y-axis: Russia (pre/war)
      - Δ labels per country, value labels on bars
    """

    #period windows
    g_pre = df[(df[col_entity].str.contains("Germany", na=False)) & (df[col_year].between(1930, 1932))][col_rol].mean()
    g_war = df[(df[col_entity].str.contains("Germany", na=False)) & (df[col_year].between(1939, 1945))][col_rol].mean()
    r_pre = df[(df[col_entity] == "Russia") & (df[col_year].between(2010, 2018))][col_rol].mean()
    r_war = df[(df[col_entity] == "Russia") & (df[col_year].between(2022, 2024))][col_rol].mean()

    #x positions: 0 for Germany, 1 for Russia
    x = np.array([0, 1], dtype=float)
    width = 0.42

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()  # second y-axis for Russia

    #bars
    #germany on LEFT axis at x=0 (two bars: pre and war)
    b_g_pre = axL.bar(x[0] - width/2, g_pre, width, color="red", alpha=0.5)
    b_g_war = axL.bar(x[0] + width/2, g_war, width, color="red", alpha=1.0)

    #russia on RIGHT axis at x=1 (two bars: pre and war)
    b_r_pre = axR.bar(x[1] - width/2, r_pre, width, color="black", alpha=0.5)
    b_r_war = axR.bar(x[1] + width/2, r_war, width, color="black", alpha=1.0)

    #titles/axis
    axL.set_title("Rule of Law — Pre vs War Period Averages")
    axL.set_ylabel("Germany — Rule of Law Index")
    axR.set_ylabel("Russia — Rule of Law Index")
    axL.set_xticks(x)
    axL.set_xticklabels(["Germany", "Russia"])
    axL.grid(False)
    axR.grid(False)

    #legend 
    legend_handles = [
        Patch(facecolor="red", alpha=0.5, label="Germany: Pre-war"),
        Patch(facecolor="red", alpha=1.0, label="Germany: War"),
        Patch(facecolor="black", alpha=0.5, label="Russia: Pre-war"),
        Patch(facecolor="black", alpha=1.0, label="Russia: War"),
    ]
    axL.legend(handles=legend_handles, loc="best")

    #helpers
    def _bar_height(container):
        if container and len(container.patches) > 0:
            h = container.patches[0].get_height()
            try:
                return float(h)
            except Exception:
                return np.nan
        return np.nan

    def _annotate_values(ax, container):
        h = _bar_height(container)
        if not np.isnan(h):
            rect = container.patches[0]
            ax.text(rect.get_x() + rect.get_width()/2, h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # Δ labels: compute per country on its own axis scale and place between the two bars
    def _delta_label(ax, cont_pre, cont_war, label_country):
        y0 = _bar_height(cont_pre)
        y1 = _bar_height(cont_war)
        if np.isnan(y0) or np.isnan(y1):
            return
        # midpoint x between the two bars
        x_mid = (cont_pre.patches[0].get_x() + cont_pre.patches[0].get_width()/2 +
                 cont_war.patches[0].get_x() + cont_war.patches[0].get_width()/2) / 2
        # place at mid-height of the two values on THIS axis
        y_mid = (y0 + y1) / 2.0
        ax.text(x_mid, y_mid, f"Δ {label_country}: {y1 - y0:+.2f}",
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    #apply annotations
    _annotate_values(axL, b_g_pre)
    _annotate_values(axL, b_g_war)
    _annotate_values(axR, b_r_pre)
    _annotate_values(axR, b_r_war)

    _delta_label(axL, b_g_pre, b_g_war, "Germany")
    _delta_label(axR, b_r_pre, b_r_war, "Russia")

    #tight layout & save
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


#FIGURE 4: Grouped bar chart pre vs war averages with Δ labels
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

    ax.set_title("Rule of Law — Pre vs War Period Averages")
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
    fig3a_since_regime_dual_axes_pct(df, col_entity, col_year, col_rol, outpath=outdir / "fig3a_since_regime_dual_pct.png")
    fig4_grouped(df, col_entity, col_year, col_rol, outdir / "fig4_grouped_pre_vs_war.png")
    fig4a_dual_axis(df, col_entity, col_year, col_rol, outdir / "fig4a_dual_axis.png")


    print("Saved:")
    print(outdir / "fig1_germany_1930_1950.png")
    print(outdir / "fig2_russia_1999_2024.png")
    print(outdir / "fig3_since_regime_start_topbaseline.png")
    print(outdir / "fig3a_since_regime_dual_pct.png")
    print(outdir / "fig4_grouped_pre_vs_war.png")
    print(outdir / "fig4a_dual_axis.png")

if __name__ == "__main__":
    main()