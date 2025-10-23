import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  #non-interactive backend for scripts

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent

from lab07_data_fitting.src.data_fitting import (
    MODELS, fit_model, mle_fit_model, 
    correlation_coefficient, 
    chi_square, reduced_chi_square,
    aic_rss, bic_rss, aic_from_ll, bic_from_ll,
    plot_residuals, plot_fit
) 

#robust data/output paths
DATA = PKG_ROOT / "data" / "us_election.csv"
OUT = PKG_ROOT /"output"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA, sep=";")

#clean and select numeric columns
fv = df["fraction_votes"].astype(str).str.replace("%","",regex=False).str.strip()
fv = pd.to_numeric(fv, errors="coerce")
if fv.max(skipna=True) is not None and fv.max() > 1.5:
    fv = fv / 100.0
df["fraction_votes"] = fv

df = df.dropna(subset=["votes", "fraction_votes"])
df = df[(df["votes"] >= 0) & (df["fraction_votes"] >= 0) & (df["fraction_votes"] <= 1)] 

#choose vars for fitting
x = df["votes"].astype(float).values
y = df["fraction_votes"].values

#scale x for better fitting
x_scaled = (x - x.mean()) / (x.std() + 1e-12)

#fit and evaluate each model
models_to_run = ["linear", "quadratic", "exponential"]
rows = []

for model in models_to_run:
    k = MODELS[model][2]

    #OLS via curve_fit
    try:
        y_hat_ols, params_ols = fit_model(x_scaled, y, model=model)
        r = correlation_coefficient(x_scaled, y)
        chi2 = chi_square(y, y_hat_ols)
        red = reduced_chi_square(chi2, n=len(y), k=k)
        AIC_rss_val = aic_rss(y, y_hat_ols, k=k)
        BIC_rss_val = bic_rss(y, y_hat_ols, k=k)
    
        #persist plots
        plot_fit(
             x_scaled, y, y_hat_ols, model_name=f"{model}_ols", 
             save_path=str(OUT / f"{model}_ols_fit.png")
        )
        plot_residuals(
            y, y_hat_ols,
            save_path=str(OUT / f"{model}_ols_residuals.png")
        )
    except Exception as e:
        print(f"Error fitting model {model}: {e}")
        y_hat_ols, params_ols = np.nan, np.nan
        r = chi2 = red = AIC_rss_val = BIC_rss_val = np.nan

    #MLE via log-likelihood
    try:
            y_hat_mle, params_mle, sigma_hat, ll, nll = mle_fit_model(x_scaled, y, model=model)
            
            #AIC/BIC using full log-likelihood; include sigma as a parameter -> k+1
            AIC_mle = aic_from_ll(ll, k=k+1)
            BIC_mle = bic_from_ll(ll, k=k+1, n=len(y))
            
            #persist plots for the MLE fit
            plot_fit(
                x_scaled, y, y_hat_mle, model_name=f"{model}_mle",
                save_path=str(OUT / f"{model}_mle_fit.png")
            )
            plot_residuals(
                y, y_hat_mle,
                save_path=str(OUT / f"{model}_mle_residuals.png")
            )
    except Exception as e:
        print(f"Error fitting model {model} (MLE): {e}")
        y_hat_mle = params_mle = sigma_hat = ll = nll = AIC_mle = BIC_mle = np.nan

    #collect results
    rows.append({
        "model": model,
        "params_ols": params_ols,
        "AIC_rss": None if isinstance(AIC_rss_val, float) and np.isnan(AIC_rss_val) else AIC_rss_val,
        "BIC_rss": None if isinstance(BIC_rss_val, float) and np.isnan(BIC_rss_val) else BIC_rss_val,
        "params_mle": params_mle,
        "sigma_hat": sigma_hat,
        "loglik": None if isinstance(ll, float) and np.isnan(ll) else ll,
        "AIC_mle": None if isinstance(AIC_mle, float) and np.isnan(AIC_mle) else AIC_mle,
        "BIC_mle": None if isinstance(BIC_mle, float) and np.isnan(BIC_mle) else BIC_mle,
        "r_linear_x_y": None if isinstance(r, float) and np.isnan(r) else r,
        "chi2_ols": None if isinstance(chi2, float) and np.isnan(chi2) else chi2,
        "red_chi2_ols": None if isinstance(red, float) and np.isnan(red) else red,
    })

summary = pd.DataFrame(rows)

#order by AIC_mle (if available), then BIC_mle
order_cols = [c for c in ["AIC_mle", "BIC_mle", "AIC_rss", "BIC_rss"] if c in summary.columns]
summary = summary.sort_values(by=order_cols, ascending=True)
print("\n=== Model comparison (MLE vs OLS) ===")
print(summary.to_string(index=False))