#script executing full pipeline
#reproducible - CI can run one command and get CSV + plot + results

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

#import pure functions + config 
from lab04_data_pipeline.src.line_pipeline import (
    GenConfig, Robust, generate_data, fit_line, save_plot,
    generate_data_quadratic, fit_quadratic_ols, generate_data_exponential, fit_exponential_mle, r2_score, aic_from_rss,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run linear model pipeline.")
    #model family
    p.add_argument("--model", choices=["line", "quadratic", "exponential"], default="line", help="model family")
    
  # common (line+quad+exp) base parameters
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x-start", type=float, default=0.0)
    p.add_argument("--x-stop", type=float, default=10.0)
    p.add_argument("--noise", type=float, default=1.0, help="noise sigma (linear/quad); see --sigma-ln for exp")
    p.add_argument("--outdir", type=Path, default=Path("artifacts"))
    p.add_argument("--write-meta", action="store_true", help="write meta.json (line model only)")

    # line model params & options
    p.add_argument("--m", type=float, default=2.5, help="true slope (line) or 'a' in exp")
    p.add_argument("--b", type=float, default=1.0, help="true intercept (line) or 'b' in exp")
    p.add_argument("--heteroscedastic", action="store_true")
    p.add_argument("--sigma-min", type=float, default=0.3)
    p.add_argument("--sigma-max", type=float, default=1.8)
    p.add_argument("--n-outliers", "--n_outliers", dest="n_outliers", type=int, default=0)
    p.add_argument("--wls", action="store_true")
    p.add_argument("--robust", choices=["none", "huber"], default="none")

    # quadratic params
    p.add_argument("--quad-a", type=float, default=0.3)
    p.add_argument("--quad-b", type=float, default=-1.2)
    p.add_argument("--quad-c", type=float, default=2.0)

    # exponential params
    p.add_argument("--sigma-ln", type=float, default=0.15, help="log-space sigma for exp model")

    return p.parse_args()

def main() -> None:
    args = parse_args()

    robust = Robust.HUBER if args.robust == "huber" else Robust.NONE

    if args.model == "line":
        #build config
        meta_dest = Optional[Path] = (args.outdir / "meta.json") if args.write_meta else None
        cfg = GenConfig(
            m=args.m, b=args.b, n=args.n,
            x_start=args.x_start, x_stop=args.x_stop,
            heteroscedastic=args.heteroscedastic, 
            noise_sigma=args.noise,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            n_outliers=args.n_outliers, seed=args.seed,
            outdir=args.outdir,
            csv_path=args.outdir / "data.csv", 
            meta_path=meta_dest, 
            plot_path=args.outdir / "plot.png",
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
        # 6) Results json
        out_dir = Path("lab04_data_pipeline/outputs"); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps({
            "csv": str(csv_p), "plot": str(plot_p),
            "rmse": rmse, "r2": r2, "m_est": m_est, "b_est": b_est,
            "true_m": m_true, "true_b": b_true        
        }, indent=2))
        print(f"RESULTS: {out_dir / 'results.json'}")

    elif args.model == "quadratic":
        # 1) Generate data
        csv_p, _ = generate_data_quadratic(
            a=args.quad_a, b=args.quad_b, c=args.quad_c,
            n=args.n, x_min=args.x_start, x_max=args.x_stop,
            sigma=args.noise, seed=args.seed,
            heteroscedastic=args.heteroscedastic,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            n_outliers=args.n_outliers, outdir=args.outdir
        )
        print(f"Generated data CSV: {csv_p}")

        # 2) Fit
        a_hat, b_hat, c_hat = fit_quadratic_ols(csv_p)

        # 3) Metrics + plot
        df = pd.read_csv(csv_p); x = df["x"].to_numpy(); y = df["y"].to_numpy()
        y_hat = a_hat * x**2 + b_hat * x + c_hat
        rss = float(np.sum((y - y_hat) ** 2))
        print(f"True quad: a={args.quad_a:.3f}, b={args.quad_b:.3f}, c={args.quad_c:.3f}")
        print(f"Fit  quad: a={a_hat:.3f}, b={b_hat:.3f}, c={c_hat:.3f}")
        print(f"R^2: {r2_score(y, y_hat):.4f} | AIC: {aic_from_rss(len(x), k=3, rss=rss):.2f}")

        xx = np.linspace(x.min(), x.max(), 200)
        yy_true = args.quad_a * xx**2 + args.quad_b * xx + args.quad_c
        yy_fit  = a_hat * xx**2 + b_hat * xx + c_hat
        curve_png = args.outdir / "plot.png"
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, s=18, alpha=0.8, label="data")
        plt.plot(xx, yy_true, linewidth=2, label="True curve")
        plt.plot(xx, yy_fit, linewidth=2, linestyle="--", label="Fitted curve")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Quadratic fit: true vs fitted")
        plt.legend(); plt.tight_layout(); plt.savefig(str(curve_png), dpi=150); plt.close()
        print(f"Saved plot to: {curve_png}")

    else:  # args.model == "exp"
        # 1) Generate data (homoscedastic log-noise by default)
        csv_p, _ = generate_data_exponential(
            a=args.m, b=args.b, n=args.n,
            x_min=args.x_start, x_max=args.x_stop,
            sigma_ln=args.sigma_ln, seed=args.seed, outdir=args.outdir
        )
        print(f"Generated data CSV: {csv_p}")

        # 2) Fit (MLE via log-linear OLS)
        a_hat, b_hat = fit_exponential_mle(csv_p)

        # 3) Metrics + plot
        df = pd.read_csv(csv_p); x = df["x"].to_numpy(); y = df["y"].to_numpy()
        y_hat = a_hat * np.exp(b_hat * x)
        rss = float(np.sum((y - y_hat) ** 2))
        print(f"True exp: a={args.m:.3f}, b={args.b:.3f}")
        print(f"Fit  exp: a={a_hat:.3f}, b={b_hat:.3f}")
        print(f"R^2: {r2_score(y, y_hat):.4f} | AIC: {aic_from_rss(len(x), k=2, rss=rss):.2f}")

        xx = np.linspace(x.min(), x.max(), 200)
        exp_png = args.outdir / "plot.png"
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, s=18, alpha=0.8, label="data")
        plt.plot(xx, args.m * np.exp(args.b * xx), linewidth=2, label="True curve")
        plt.plot(xx, a_hat * np.exp(b_hat * xx), linewidth=2, linestyle="--", label="Fitted curve")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Exponential fit: true vs fitted")
        plt.legend(); plt.tight_layout(); plt.savefig(str(exp_png), dpi=150); plt.close()
        print(f"Saved plot to: {exp_png}")
    

if __name__ == "__main__":
    main()
