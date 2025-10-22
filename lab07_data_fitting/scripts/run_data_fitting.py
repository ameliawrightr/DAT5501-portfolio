import pandas as pd
import numpy as np
from lab07_data_fitting.src.data_fitting import (
    MODELS, fit_model, 
    correlation_coefficient, 
    chi_square, reduced_chi_square,
    plot_fit, aic, bic, rss,
    plot_residuals,
) 

#load US election dataset
df = pd.read_csv("lab07_data_fitting/data/us_election.csv", sep=";")

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

summary_rows = []

for model in models_to_run:
    try:
        y_hat, params = fit_model(x_scaled, y, model=model)
    except Exception as e:
        print(f"Error fitting model {model}: {e}")
        summary_rows.append({
            "model": model,
            "params": "_",
            "r": np.nan,
            "chi2": np.nan,
            "reduced_chi2": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "rss": np.nan,
            "error": str(e)
        })
        continue

    #metrics
    r = correlation_coefficient(x_scaled, y)
    chi2 = chi_square(y, y_hat)
    red_chi2 = reduced_chi_square(chi2, n=len(y), k=len(params))
    aic_val = aic(y, y_hat, k=len(params))
    bic_val = bic(y, y_hat, k=len(params))
    rss_val = rss(y, y_hat)

    #persist plots
    plot_fit(x, y, y_hat, model_name=model, 
             save_path=f"lab07_data_fitting/output/{model}_fit.png")
    plot_residuals(y, y_hat,
            save_path=f"lab07_data_fitting/output/{model}_residuals.png")   
    
    #record summary
    summary_rows.append({
        "model": model,
        "params": tuple(np.round(params, 6)),
        "r": round(r, 3),
        "chi2": round(chi2, 3),
        "reduced_chi2": round(red_chi2, 3),
        "aic": round(aic_val, 2),
        "bic": round(bic_val, 2),
        "rss": round(rss_val, 2)
    })

#print summary table
summary_df = pd.DataFrame(summary_rows)
#sort: lower AIC/BIC better, if tie lower reduced chi2 better
summary_df = summary_df.sort_values(by=["aic", "bic", "reduced_chi2"], ascending=[True, True, True])
print("\nModel Fitting Summary (lower AIC/BIC/reduced_chi2 better):")
print(summary_df.to_string(index=False))