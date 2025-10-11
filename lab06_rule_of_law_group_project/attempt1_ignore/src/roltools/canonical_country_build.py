import numpy as np
import pandas as pd

def canonical_series(rol: pd.DataFrame, rol_col:str, *, spec: dict) -> pd.DataFrame:
    """ 
    Build a canonical time series per 'spec' with either:
      - 'prefer' list: take the first entity with a non-NaN value each year
      - or 'combine': map of alias -> weight, and we take a weighted average for overlapping years
    Returns df with columns ['Year', rol_col] for the canonical country.
    """
    years = pd.Index(sorted(rol['Year'].unique()), name="Year")
    out = pd.DataFrame(index=years).reset_index()
    vals = pd.Series(np.nan, index=years, name=rol_col)

    if 'prefer' in spec:
        for entity in spec['prefer']:
            s = rol.loc[rol['Entity'] == entity, ['Year', rol_col]].set_index("Year")[rol_col]
            vals = vals.where(vals.notna(), s) #fill missing w source
    elif 'combine' in spec:
        acc = pd.Series(0.0, index=years)
        wsum = pd.Series(0.0, index=years)
        for entity, weight in spec['combine'].items():
            s = rol.loc[rol['Entity'] == entity, ['Year', rol_col]].set_index("Year")[rol_col]
            mask = s.notna()
            acc.loc[mask.index] = acc.loc[mask.index].add(s.fillna(0) * weight, fill_value=0.0)
            wsum.loc[mask.index] = wsum.loc[mask.index].add((mask.astype(float) * weight), fill_value=0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            vals = acc / wsum.replace(0, np.nan) #avoid div 0
    else:
        raise ValueError("spec must contain either 'prefer' or 'combine'")
    
    out[rol_col] = vals.values
    return out

def germany_canonical(rol: pd.DataFrame, col:str) -> pd.DataFrame:
    """ Germany canonical series 1871-2020
        Prefer: Germany if present (pre45 and post 90)
        Split period (West Germany, East Germany) 1949-1990, take simple average
    """
    #base Germany;
    g_base = (rol.loc[rol["Entity"] == "Germany", ["Year", col]]
              .groupby("Year", as_index=False)[col].mean())
    
    #splice in split period
    g_split = (rol.loc[rol["Entity"].isin(["West Germany", "East Germany"]), ["Entity", "Year", col]]
               .pivot_table(index="Year", columns="Entity", values=col, aggfunc="mean"))
    g_split["split_mean"] = g_split.mean(axis=1, skipna=True)
    g_split = g_split.reset_index()[["Year", "split_mean"]].rename(columns={"split_mean": col})

    #combine
    g_all = pd.merge(g_base, g_split, on="Year", how="outer", suffixes=("", "_split"))
    if f"{col}_split" in g_all.columns:
        g_all[col] = g_all[col].where(g_all[col].notna(), g_all[f"{col}_split"])
        g_all = g_all.drop(columns=[f"{col}_split"])

    return g_all.sort_values


def russia_canonical(rol: pd.DataFrame, col: str) -> pd.DataFrame:
    # Accept common aliases if they appear in other sources
    aliases = ["Russia", "Russian Federation", "Soviet Union"]
    r = (rol.loc[rol["Entity"].isin(aliases), ["Year", col]]
           .groupby("Year", as_index=False)[col].mean()
           .sort_values("Year").reset_index(drop=True))
    return r


