from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class GenConfig:
    # data generation parameters
    m: float = 2.5
    b: float = 1.0
    n: int = 200
    x_start: float = 0.0
    x_stop: float = 10.0
    noise_sigma: float = 1.0
    seed: int = 42
    #file paths - relative to repo root
    csv_path: str = "lab04_data_pipeline/data/synth.csv"
    meta_path: str = "lab04_data_pipeline/data/meta.json"
    plot_path: str = "lab04_data_pipeline/outputs/line_fit.png"

def generate_data(cfg: GenConfig): 
    # generate synthetic data (x, y) for a given m,b with noise & save to CSV + meta JSON
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(cfg.x_start, cfg.x_stop, cfg.n)
    y_true = cfg.m * x + cfg.b
    y = y_true + rng.normal(0, cfg.noise_sigma, size=cfg.n)

    #save x, y to CSV; save cfg to JSON
    csv_p = Path(cfg.csv_path); csv_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p = Path(cfg.meta_path); meta_p.parent.mkdir(parents=True, exist_ok=True)

    #return (csv_path, meta_path) for testing
    pd.DataFrame({"x": x, "y": y}).to_csv(csv_p, index=False)
    meta = {"m": cfg.m, "b": cfg.b, "n": cfg.n, "noise_sigma": cfg.noise_sigma,
            "x_start": cfg.x_start, "x_stop": cfg.x_stop, "seed": cfg.seed}
    meta_p.write_text(json.dumps(meta, indent=2))
    return csv_p, meta_p

def fit_line(csv_path: str):
    #load CSV, ensure numeric/no-NaN, fit y = m*x + b (least squares)
    #return (m_est, b_est, x_array, y_array) 
    df = pd.read_csv(csv_path)
    #enforce numeric & no NaNs
    df = df.apply(pd.to_numeric, errors="raise")
    if df[["x","y"]].isna().any().any():
        raise ValueError("NaN values detected in input data.")
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    m_est, b_est = np.polyfit(x, y, 1) #degree=1 polynomial (line)
    return float(m_est), float(b_est), x, y

def save_plot(x, y, true_m, true_b, est_m, est_b, out_path: str):
    #save a PNG with scatter, true line, and best-fit line
    #return out_path 
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(x.min(), x.max(), 200)
    y_true = true_m * xs + true_b
    y_est  = est_m * xs + est_b

    plt.figure()
    plt.scatter(x, y, s=12, label="Data (noisy)")
    plt.plot(xs, y_true, linewidth=2, label="True line")
    plt.plot(xs, y_est, linewidth=2, linestyle="--", label="Best fit")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()
    return out


