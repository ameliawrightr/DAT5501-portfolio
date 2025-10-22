import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

#models
def linear(x, m, c):
    return m * x + c

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b):
    return a * np.exp(b * x)

MODELS = {
    "linear": (linear, 2),
    "quadratic": (quadratic, 3),
    "exponential": (exponential, 2)
}

#fitting
def initial_guess(model_name, x, y):
    if model_name == "exponential":
        #only strictly positive y values for initial guess
        y_pos = y[y > 0]
        x_pos = x[y > 0]
        if len(y_pos) >= 2:
            A0 = max(np.median(y_pos), 1e-6)
            dx = x_pos.max() - x_pos.min()
            dy = np.log(y_pos.max()) - np.log(max(y_pos.min(), 1e-12))
            B0 = dy / dx if dx != 0 else 0
            return (A0, B0)
        return [max(np.mean(y), 1e-6), 0.0]
    #lin/quad: let curve_fit handle it
    return None

def fit_model(x, y, model="linear"):
    #fit model to (x,y)
    #returns fitted y values and optimal parameters

    if model not in MODELS:
        raise ValueError(f"Model '{model}' not recognized. Choose from {list(MODELS.keys())}.")
    
    fn, k = MODELS[model]

    #exp needs strictly positive y for curve_fit stability
    mask = np.ones_like(y, dtype=bool)
    if model == "exponential":
        mask = y > 0
        if mask.sum() < 3:
            raise RuntimeError("Not enough positive y values for exponential fitting.")

    p0 = initial_guess(model, x[mask], y[mask])
    popt, _ = curve_fit(fn, x[mask], y[mask], p0=p0, maxfev=20000)

    y_fit = fn(x, *popt)
    return y_fit, tuple(popt)

#stats
def correlation_coefficient(x, y):
    return float(np.corrcoef(x, y)[0, 1])

def chi_square(y_obs, y_fit, sigma=None):
    y_obs = np.asarray(y_obs).ravel()
    y_fit = np.asarray(y_fit).ravel()
    if y_obs.shape != y_fit.shape:
        raise ValueError(f"Shape mismatch: y_obs {y_obs.shape} vs y_fit {y_fit.shape}")
    if sigma is None:
        sigma = np.ones_like(y_obs)
    else:
        sigma = np.asarray(sigma).ravel()
        if sigma.shape != y_obs.shape:
            raise ValueError(f"Shape mismatch: sigma {sigma.shape} vs y_obs {y_obs.shape}")
    return float(np.sum(((y_obs - y_fit) / sigma) ** 2))

def reduced_chi_square(chi2, n, k):
    #n=observaions, k=parameters
    dof = max(n - k, 1)
    return float(chi2 / dof)

def rss(y_obs, y_fit):
    e = np.asarray(y_obs) - np.asarray(y_fit)
    return float(np.sum(e**2))

def aic(y_obs, y_fit, k):
    #gaussian likelihodd w unknown var
    n = len(y_obs)
    resid = rss(y_obs, y_fit)
    return float(n * np.log(resid / n + 1e-12) + 2 * k)

def bic(y_obs, y_fit, k):
    n = len(y_obs)
    resid = rss(y_obs, y_fit)
    return float(n * np.log(resid / n + 1e-12) + k * np.log(n + 1e-12))


#plot
def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def plot_fit(x, y, y_fit, model_name, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Data', color='blue')

    #sort by x for clean line
    order = np.argsort(x)

    plt.plot(x[order], np.asarray(y_fit)[order], label=f'Fitted {model_name} model', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Data Fitting using {model_name} Model')
    plt.legend()
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_residuals(y, y_fit, save_path=None):
    res = np.asarray(y) - np.asarray(y_fit)
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(res)), res, s=6)
    plt.axhline(0)
    plt.xlabel("Observation index")
    plt.ylabel("Residual (y - y_fit)")
    plt.title("Residuals")
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Residual plot saved to: {save_path}")
    else:
        plt.show()