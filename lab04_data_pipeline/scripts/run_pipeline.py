#script executing full pipeline
#reproducible - CI can run one command and get CSV + plot + results

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

#import pure functions + config 
from lab04_data_pipeline.src.line_pipeline import (
    GenConfig, Robust, generate_data, fit_line, save_plot 
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run linear model pipeline.")
    # True parameters and data gen
    p.add_argument("--m", type=float, default=2.5, help="true slope")
    p.add_argument("--b", type=float, default=1.0, help="true intercept")
    p.add_argument("--n", type=int, default=200, help="number of points")
    p.add_argument("--noise", type=float, default=1.0, help="Gaussian noise sigma")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--x-start", type=float, default=0.0, help="x range start")
    p.add_argument("--x-stop", type=float, default=10.0, help="x range stop")

    # Heteroscedastic + outliers
    p.add_argument("--heteroscedastic", action="store_true", help="use per-point sigma_y")
    p.add_argument("--sigma-min", type=float, default=0.3, help="min sigma_y if heteroscedastic")
    p.add_argument("--sigma-max", type=float, default=1.8, help="max sigma_y if heteroscedastic")
    p.add_argument("--n-outliers", "--n_outliers", dest="n_outliers", type=int, default=0, help="number of large outliers to inject")

    # Fit controls
    p.add_argument("--wls", action="store_true", help="weighted least squares (requires sigma_y)")
    p.add_argument("--robust", choices=["none", "huber"], default="none", help="robust fitter")

    # IO
    p.add_argument("--outdir", type=Path, default=Path("artifacts"), help="output directory")
    p.add_argument("--write-meta", action="store_true", help="also write meta.json next to CSV")

    return p.parse_args()

def main() -> None:
    args = parse_args()

    robust = Robust.HUBER if args.robust == "huber" else Robust.NONE

    #build config
    meta_dest = (args.outdir / "meta.json") if args.write_meta else None
    cfg = GenConfig(
        m=args.m, b=args.b, n=args.n,
        x_start=args.x_start, x_stop=args.x_stop,
        heteroscedastic=args.heteroscedastic, noise_sigma=args.noise,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        n_outliers=args.n_outliers, seed=args.seed,
        outdir=args.outdir,
        csv_path=args.outdir / "data.csv", meta_path=meta_dest, plot_path=args.outdir / "plot.png",
    )

    #1. generate data -> return paths
    csv_p, meta_p = generate_data(cfg)
    print(f"Generated data CSV: {csv_p}, meta JSON: {meta_p if meta_p else 'None'}")

    #2. fit the model (fit_line accepts path and returns m, b, x, y)
    m_est, b_est, x, y = fit_line(csv_p, use_wls=args.wls, robust=robust)

    #3. determine paramaters to report
    if meta_p:
        meta = json.loads(Path(meta_p).read_text())
        m_true = float(meta["m"])
        b_true = float(meta["b"])
    else:
        m_true = cfg.m
        b_true = cfg.b

    #extra metrics 
    #predictions from fitted line
    y_hat = m_est * x + b_est
    #RMSE = sqrt(mean(residuals^2))
    rmse = float(np.sqrt(np.mean((y-y_hat)**2)))
    #R^2 = 1 - SSR/SST
    r2 = float(1.0 - np.sum((y-y_hat)**2)/(np.sum((y-np.mean(y))**2)+1e-12))

    print(f"Fitted line: y = {m_est:.3f}x + {b_est:.3f}")
    print(f"RMSE: {rmse:.4f} | R^2: {r2:.4f}")

    #5. save plot (scatter, true line, best-fit line)
    plot_p = save_plot(x, y, m_true, b_true, m_est, b_est, out_path=cfg.plot_path, cfg=cfg)
    print(f"Saved plot to: {plot_p}")

    #6. save results JSON (true vs estimated parameters)
    #ensure output folder exists
    results_dir = Path("lab04_data_pipeline/outputs")
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "csv": str(csv_p),
        "plot": str(plot_p),
        "rmse": rmse,
        "r2": r2,
        "m_est": m_est,
        "b_est": b_est,
        "true_m": m_true,
        "true_b": b_true
    }
    (Path(results_dir) / "results.json").write_text(json.dumps(results, indent=2))

    #7. log paths (see where files went in local and CI logs)
    print(f"CSV: {csv_p}")
    print(f"PLOT: {plot_p}")
    print(f"RESULTS: {results_dir/'results.json'}")

if __name__ == "__main__":
    main()
