from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from enum import Enum
from typing import Optional, Union

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

    csv_path: Optional[Union[str, Path]] = None #if set, save CSV here
    meta_path: Optional[Union[str, Path]] = None #if set, save JSON
    plot_path: Optional[Union[str, Path]] = None #if set, save PNG here


def generate_data(cfg: GenConfig) -> tuple[Path, Optional[Path]]: 
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
    
    #always ensure outdir exists
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    #resolve output paths
    out_csv = Path(cfg.csv_path) if cfg.csv_path is not None else (cfg.outdir / "data.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_meta = None
    if cfg.meta_path is not None:
        out_meta = Path(cfg.meta_path)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "m": cfg.m, "b": cfg.b, "n": cfg.n,
            "noise_sigma": cfg.noise_sigma,
            "heteroscedastic": cfg.heteroscedastic,
            "sigma_min": cfg.sigma_min, "sigma_max": cfg.sigma_max,
            "x_start": cfg.x_start, "x_stop": cfg.x_stop,
            "n_outliers": cfg.n_outliers, "seed": cfg.seed
        }
        out_meta.write_text(json.dumps(meta, indent=2))

    return out_csv, out_meta
                                   

def fit_line(
        data: Union[pd.DataFrame, str, Path],
        use_wls: bool = False, 
        robust: Robust = Robust.NONE,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    #fit line to data in df (x, y, optional sigma_y)
    # - use_wls=True, use weighted least squares (requires sigma_y column)
    # - robust=Huber defends against outliers
    #load CSV, ensure numeric/no-NaN, fit y = m*x + b (least squares)
    #return (m_est: float, b_est: float, x: np.ndarray, y: np.ndarray) 
    
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data

    if robust == Robust.HUBER:
        m_hat, b_hat = fit_line_huber(df)
    elif use_wls:
        m_hat, b_hat = fit_line_wls(df)
    else:
        m_hat, b_hat = fit_line_ols(df)
    
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    return float(m_hat), float(b_hat), x, y


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
    w = 1.0 / (s ** 2) #weights = 1/sigma_y^2

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
        beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        m_new, b_new = float(beta[0]), float(beta[1])

        #check convergence
        if np.allclose([m_new, b_new], [m_hat, b_hat], atol=1e-6, rtol=0):
            break

        m_hat, b_hat = m_new, b_new

    return float(m_hat), float(b_hat)

def save_plot(
        x_or_df: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        m_true: float = 0.0, b_true: float = 0.0, 
        m_fit: float = 0.0, b_fit: float = 0.0, 
        out_png: Optional[Union[str, Path]]=None, 
        cfg: Optional[GenConfig] = None,
        out_path: Optional[Union[str, Path]] = None,
    ) -> Path:

    #normalise inputs - build df from (x, y)
    if isinstance(x_or_df, pd.DataFrame):
        df = x_or_df
    else:
        if y is None:
            raise ValueError("y must be provided when x is ndarray.")
        df = pd.DataFrame({"x": x_or_df, "y": y})

    #resolve out path priority: explicit arg > cfg.plot_path > outdir/plot.png
    target = out_png if out_png is not None else out_path
    if target is None:
        if cfg is not None and cfg.plot_path is not None:
            target = Path(cfg.plot_path)
        elif cfg is not None:
            target = cfg.outdir / "plot.png"
        else:
            target = Path("plot.png")
    target = Path(target)

    #save a PNG with scatter, true line, and best-fit line
    #return saved PNG
    target.parent.mkdir(parents=True, exist_ok=True)
    x_line = np.linspace(df["x"].min(), df["x"].max(), 200)
    y_true = m_true * x_line + b_true
    y_fit = m_fit * x_line + b_fit

    #matplotlib: save file (no GUI - for CI compatibility)
    plt.figure(figsize=(7,5))
    if "sigma_y" in df.columns:
        #errorbar uses vertical error bars for y
        plt.errorbar(df["x"], df["y"], yerr=df["sigma_y"], fmt='o', markersize=3, alpha=0.8, label="data ±σ")
    else:
        plt.scatter(df["x"], df["y"], s=18, alpha=0.8, label="data")
    
    #plt.scatter(x, y, s=12, label="Data (noisy)")
    plt.plot(x_line, y_true, linewidth=2, label="True line")
    plt.plot(x_line, y_fit, linewidth=2, linestyle="--", label="Best fit")
    plt.xlabel("x"); plt.ylabel("y"); plt.title("Line fit: true vs fitted")
    plt.legend(); plt.tight_layout()
    plt.savefig(str(target), dpi=150)
    plt.close()
    
    return target

def clean_nonfinite(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    #remove rows with non-finite values (NaN, inf, -inf)
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(df[c].to_numpy())
    return df.loc[mask].reset_index(drop=True)
