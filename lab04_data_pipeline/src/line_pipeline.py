from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

#dataclass - keeps parameters in one place 
# - ensures reproducibility + fewer mistakes
# - lets tests override some fields (e.g., tmp paths, diff noise)

class Robust(Enum):
    NONE = "none"
    HUBER = "huber"
    RANSAC = "ransac"

#extend GenConfig to create per point error bars and outliers
@dataclass (frozen=True)
class GenConfig:
    # data generation parameters
    m: float = 2.5 #true slope
    b: float = 1.0 #true intercept
    n: int = 200 #number of points
    x_start: float = 0.0 #x range start
    x_stop: float = 10.0 #x range stop
    #if heteroscedastic=True, each point gets own sigma_y[i] and noise - N(0, sigma_y[i])
    heteroscedastic: bool = False
    noise_sigma: float = 1.0 #Gaussian noise stddev
    #controls heteroscedastic range (sigma in [sigma_min, sigma_max])
    sigma_min: float = 0.3
    sigma_max: float = 1.8
    #n_outliers with large dev test robustness
    n_outliers: int = 0
    seed: int = 42 #RNG seed to make runs reproducible
    outdir: Path = Path("artifacts")

    #output file locations - relative to repo root
    csv_path: str = "lab04_data_pipeline/data/synth.csv"
    meta_path: str = "lab04_data_pipeline/data/meta.json"
    plot_path: str = "lab04_data_pipeline/outputs/line_fit.png"

def generate_data(cfg: GenConfig) -> pd.DataFrame: 
    #generate synthetic data from true line - y = m*x + b with Gaussian noise
    #produces:
    # - CSV with x, y columns
    # - JSON 'meta' file with ground truth parameters
    # - produce per point sigma_y (error bars) and inject outliers
    #return:
    # - (csv_path: Path, meta_path: Path)
    # - saves data.csv - inc sigma_y if heteroscedastic=True

    #reproducible RNG
    rng = np.random.default_rng(cfg.seed)
    x = rng.uniform(cfg.x_start, cfg.x_stop, size=cfg.n)

    if cfg.heteroscedastic:
        #heteroscedastic noise: each point has own sigma_y[i]
        sigma_y = rng.uniform(cfg.sigma_min, cfg.sigma_max, size=cfg.n)
        noise = rng.normal(0, sigma_y)
    else:
        sigma_y = np.full(cfg.n, cfg.noise_sigma, dtype=float)
        noise = rng.normal(0, cfg.noise_sigma, size=cfg.n)
    
    y = cfg.m * x + cfg.b + noise

    #inject outliers to test robustness
    if cfg.n_outliers > 0:
        idx = rng.choice(cfg.n, size=min(cfg.n_outliers, cfg.n), replace=False)
        y[idx] += rng.normal(0.0, 15.0, size=len(idx)) #large dev

    df = pd.DataFrame({"x": x, "y": y})
    if cfg.heteroscedastic:
        df["sigma_y"] = sigma_y
    
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    (cfg.outdir / "data.csv").write_text(df.to_csv(index=False))
    return df
                                   
    #make evenly spaced x, compute noiseless y and noisy y
    x = np.linspace(cfg.x_start, cfg.x_stop, cfg.n)
    y_true = cfg.m * x + cfg.b
    y = y_true + rng.normal(0, cfg.noise_sigma, size=cfg.n)

    #check folders exist before writing
    csv_p = Path(cfg.csv_path); csv_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p = Path(cfg.meta_path); meta_p.parent.mkdir(parents=True, exist_ok=True)

    #write dataset to CSV
    pd.DataFrame({"x": x, "y": y}).to_csv(csv_p, index=False)

    #also save true parameters to JSON meta file
    meta = {
        "m": cfg.m, "b": cfg.b, "n": cfg.n, 
        "noise_sigma": cfg.noise_sigma,
        "x_start": cfg.x_start, "x_stop": cfg.x_stop, 
        "seed": cfg.seed
    }
    meta_p.write_text(json.dumps(meta, indent=2))

    return csv_p, meta_p

def fit_line(df: pd.DataFrame, use_wls: bool = False, robust: Robust = Robust.NONE) -> tuple[float, float]:
    #fit line to data in df (x, y, optional sigma_y)
    # - use_wls=True, use weighted least squares (requires sigma_y column)
    # - robust=Huber defends against outliers
    #load CSV, ensure numeric/no-NaN, fit y = m*x + b (least squares)
    #return (m_est: float, b_est: float, x: np.ndarray, y: np.ndarray) 
    
    #df = pd.read_csv(csv_path)

    if robust == Robust.HUBER:
        return fit_line_huber(df)
    if use_wls:
        return fit_line_wls(df)
    return fit_line_ols(df)

    #defensive checks
    # - enforce numeric & no NaNs
    df = df.apply(pd.to_numeric, errors="raise")
    if df[["x","y"]].isna().any().any():
        raise ValueError("NaN values detected in input data.")
    
    #polyfit with degree=1 (line)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    m_est, b_est = np.polyfit(x, y, 1) #degree=1

    return float(m_est), float(b_est), x, y

def fit_line_ols(df: pd.DataFrame) -> tuple[float, float]:
    #plain OLS: good when noise is homoscedastic and no outliers
    df = clean_nonfinite(df, ("x", "y"))
    m_hat, b_hat = np.polyfit(df["x"].to_numpy(), df["y"].to_numpy(), 1)
    return float(m_hat), float(b_hat)

def fit_line_wls(df: pd.DataFrame) -> tuple[float, float]:
    #weighted least squares using per point uncertainties sigma_y
    if "sigma_y" not in df.columns:
        raise ValueError("sigma_y column required for WLS.")
    df = clean_nonfinite(df, ("x", "y", "sigma_y"))
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    s = df["sigma_y"].to_numpy()
    w = 1.0 / (s ** 2) #weights = 1/s

    #solve normal equations for weighted linear regression
    X = np.stack([x, np.ones_like(x)], axis=1) #design matrix
    W = np.diag(w) #weight matrix
    #(X^T W X) beta = X^T W y
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy) 
    m_hat, b_hat = beta[0], beta[1]
    return float(m_hat), float(b_hat)

def fit_line_huber(df: pd.DataFrame, delta: float = 1.0, max_iter: int = 50) -> tuple[float, float]:
    #robust line fit using Huber loss (iteratively reweighted least squares)
    #delta: threshold between L2 and L1 loss, smaller = more robust

    df = clean_nonfinite(df, ("x", "y"))
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    X = np.stack([x, np.ones_like(x)], axis=1)

    #start from OLS solution
    m_hat, b_hat = np.polyfit(x, y, 1)

    for it in range(max_iter):
        y_pred = m_hat * x + b_hat
        r = y - y_pred #residuals
        #huber weights: w = 1 if |r| <= delta else delta/|r|
        abs_r = np.abs(r)

        #compute weights based on Huber loss
        w = np.where(abs_r <= delta, 1.0, (delta / (abs_r + 1e-12)))

        #solve weighted least squares with these weights
        X = np.stack([x, np.ones_like(x)], axis=1)
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta = np.linalg.solve(XtWX, XtWy, rcond=None)[0]
        m_new, b_new = float(beta[0]), float(beta[1])

        #check convergence
        if np.isclose(m_new, m_hat) and np.isclose(b_new, b_hat):
            break

        m_hat, b_hat = m_new, b_new

    return float(m_hat), float(b_hat)

def save_plot(df: pd.DataFrame, m_true: float, b_true: float, m_fit: float, b_fit: float, out_png: Path) -> Path:
    #save a PNG with scatter, true line, and best-fit line
    #return saved PNG
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_line = np.linspace(df["x"].min(), df["x"].max(), 200)
    y_true = m_true * x_line + b_true
    y_fit = m_fit * x_line + b_fit

    #matplotlib: save file (no GUI - for CI compatibility)
    plt.figure(figsize=(7,5))
    if "sigma_y" in df.columns:
        #errorbar uses vertical errors bars for y
        plt.errorbar(df["x"], df["y"], yerr=df["sigma_y"], fmt='o', markersize=3, alpha=0.8, label="data ±σ")
    else:
        plt.scatter(df["x"], df["y"], s=18, alpha=0.8, label="data")
    
    #plt.scatter(x, y, s=12, label="Data (noisy)")
    plt.plot(x_line, y_true, linewidth=2, label="True line")
    plt.plot(x_line, y_fit, linewidth=2, linestyle="--", label="Best fit")
    plt.xlabel("x"); plt.ylabel("y"); plt.title("Line fit: true vs fitted")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    return out_png

def clean_nonfinite(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    #remove rows with non-finite values (NaN, inf, -inf)
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(df[c].to_numpy())
    return df.loc[mask].reset_index(drop=True)
