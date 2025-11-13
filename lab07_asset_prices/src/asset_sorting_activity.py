import time
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

pkg_root = Path(__file__).resolve().parents[1]

#load prices
def load_prices(data_path: Path) -> pd.DataFrame:
    #tolerated BOMs and odd encodings
    df = pd.read_csv(data_path, encoding="utf-8-sig", thousands=",")
    df.columns = [c.strip() for c in df.columns]
    
    required = {"Date", "Close"}
    if not required .issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}. Got {list(df.columns)}")

    df["Close"] = (
        df["Close"].astype(str)
        .str.replace(r'[\$,]', '', regex=True)
        .replace({"":np.nan})
    )

    df = df[["Date", "Close"]].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    
    df = df.dropna(subset=["Date","Close"]).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 2:
        print(f"Loaded {len(df)} valid records from {data_path}")
        raise ValueError("Not enough valid data points after cleaning.")
    
    return df


#compute price change
def compute_price_change(df: pd.DataFrame) -> pd.Series:
    delta = df["Close"].diff()
    return delta.dropna()

#time sort
def time_sort(values: List[float], n_values: Iterable[int], reps: int = 5) -> List[float]:
    times_ms: List[float] = []
    for n in n_values:
        prefix = values[:n]
        acc_ns = 0
        for _ in range(reps):
            arr = prefix.copy()
            t0 = time.perf_counter_ns()
            arr.sort()
            t1 = time.perf_counter_ns()
            acc_ns += (t1 - t0)
        times_ms.append(acc_ns / reps / 1e6)  # Convert to milliseconds    
    return times_ms

#plot T(n) vs n and compare to n log n
def plot_time_complexity(ns: List[int], times_ms: List[float], outpath:Path) -> None:
    if not ns or not times_ms:
        raise ValueError("Nothing to plot: 'ns' or 'times_ms' is empty.")
    
    x = np.array(ns, dtype=float)
    nlogn = x * np.log2(np.maximum(x,2))
    scale = times_ms[-1] / nlogn[-1]
    curve = scale * nlogn

    plt.figure(figsize=(10, 6))
    plt.plot(ns, times_ms, label="Measured Sort Time (ms)")
    plt.plot(ns, curve, label="Scaled n log n", linestyle='--')
    plt.xlabel("Number of Price Changes (n)")
    plt.ylabel("Time (ms)")
    plt.title("Sorting Time Complexity of Price Changes")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

#orchestratror function
def run_end_to_end(data_path: Path, outdir: Path) -> Tuple[Path,Path,Path]:
    #run full activity and writes artifacts
    #load and compute
    df = load_prices(data_path)
    delta = compute_price_change(df)
    values = delta.tolist()

    #n from 7 to 365 or len(values)
    Nmax = min(365, len(values))
    if Nmax < 7:
        if Nmax < 2:
            raise ValueError("Not enough price change data to run sorting analysis (need at least 2).")
        ns = list(range(2, Nmax + 1))
    else:
        ns = list(range(7, Nmax + 1))
    

    times_ms = time_sort(values, ns, reps=5)

    #save timings
    outdir.mkdir(parents=True, exist_ok=True)
    timing_df = pd.DataFrame({"n": ns, "time_ms": times_ms})
    timing_csv = pkg_root / "data" / "sorting_times.csv"
    timing_df.to_csv(timing_csv, index=False)

    #save price change series
    pricechange_csv = pkg_root / "data" / "price_changes.csv"
    delta.reset_index(drop=True).to_frame("Delta_P").to_csv(pricechange_csv, index=False)
    
    #plot
    plot_path = outdir / "plots" / "price_change_sorting.png"
    plot_time_complexity(ns, times_ms, plot_path)

    return timing_csv, pricechange_csv, plot_path


