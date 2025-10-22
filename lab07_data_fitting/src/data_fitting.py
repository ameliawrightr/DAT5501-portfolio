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
    "linear": linear,
    "quadratic": quadratic,
    "exponential": exponential
}

#fitting
def fit_model(x, y, model="linear"):
    if model not in MODELS:
        raise ValueError(f"Model '{model}' not recognized. Choose from {list(MODELS.keys())}.")
    
    f = MODELS[model]
    popt, _ = curve_fit(f, x, y)
    y_fit = f(x, *popt)
    return y_fit, tuple(popt)

#stats
def correlation_coefficient(x, y):
    return np.corrcoef(x, y)[0, 1]

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

#plot
def plot_fit(x, y, y_fit, model_name, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Data', color='blue')
    plt.plot(x, y_fit, label=f'Fitted {model_name} model', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Data Fitting using {model_name} Model')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Residual plot saved to: {save_path}")
    else:
        plt.show()