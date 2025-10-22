import pandas as pd
import numpy as np
from lab07_data_fitting.src.data_fitting import (
    fit_model, 
    correlation_coefficient, 
    chi_square, 
    plot_fit,
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
  

#choose vars for fitting
x = df["votes"].astype(float).values
y = df["fraction_votes"].values

#scale x for better fitting
x_scaled = (x - x.mean()) / (x.std() + 1e-12)

#linear fit
y_fit, params = fit_model(x_scaled, y, model="linear")
m,c = params
print(f"Linear fit parameters: slope={m:.6f}, intercept={c:.6f}")

#stats
r = correlation_coefficient(x_scaled, y)
chi2 = chi_square(y, y_fit)
print(f"Correlation coefficient (r): {r:.3f}")
print(f"Goodness of fit (χ²): {chi2:.3f}")

#plots
from lab07_data_fitting.src.data_fitting import MODELS
y_hat_unscaled_axis = MODELS["linear"](x_scaled, *params)

plot_fit(x, y, y_hat_unscaled_axis, model_name="linear", 
         save_path="lab07_data_fitting/output/linear_fit.png")
plot_residuals(y, y_hat_unscaled_axis,
        save_path="lab07_data_fitting/output/linear_residuals.png")