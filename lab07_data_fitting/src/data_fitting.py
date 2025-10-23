import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit, minimize
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
    "linear": (linear, [1.0, 0.0], 2),
    "quadratic": (quadratic, [1.0, 0.0, 0.0], 3),
    "exponential": (exponential, [1.0, 0.0], 2)
}

#fitting
def finite_mask(*arrays):
    #return boolean mask where all arrays have finite values
    mask = np.ones_like(arrays[0].ravel(), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr).ravel())
    return mask

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

def fit_model(x, y, model, p0=None):
    #fit model to (x,y)
    #returns fitted y values and optimal parameters

    # ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    if model not in MODELS:
        raise ValueError(f"Model '{model}' not recognized. Choose from {list(MODELS.keys())}.")

    fn, default_p0, k = MODELS[model]

    #exp needs strictly positive y for curve_fit stability
    mask = np.ones_like(y, dtype=bool)
    if model == "exponential":
        mask = y > 0
        if mask.sum() < 3:
            raise RuntimeError("Not enough positive y values for exponential fitting.")

    # determine p0: prefer explicit p0, then heuristic, then model default
    if p0 is None:
        p0 = initial_guess(model, x[mask], y[mask])
        if p0 is None:
            p0 = default_p0

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

def aic_from_ll(ll, k):
    # AIC = 2*k - 2*ln(L)
    return float(2 * k - 2 * ll)

def bic_from_ll(ll, k, n):
    # BIC = k * ln(n) - 2 * ln(L)
    return float(k * np.log(n + 1e-12) - 2 * ll)

def aic_rss(y_obs, y_fit, k):
    #classic AIC using RSS w gaussian assumption and unknown variance
    n = len(y_obs)
    resid = rss(y_obs, y_fit)
    return float(n * np.log(resid / n + 1e-12) + 2 * k)

def bic_rss(y_obs, y_fit, k):
    #classic BIC using RSS w gaussian assumption and unknown variance
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


#MLE (Gaussian)
def nll_gaussian(theta, x, y, fn, k_params):
    #negative log likelihood for Gaussian errors
    #theta = [model params..., sigma]
    params = theta[:k_params]
    log_sigma = theta[-1]
    sigma = np.exp(log_sigma)
    y_fit = fn(x, *params)
    resid = y - y_fit
    n = y.size

    #NLL = 0.5 * sum((resid/sigma)^2) + n * log(sigma) + 0.5 * n * log(2pi)
    #constant term can be ignored in optimization
    nll = 0.5 * np.sum((resid / sigma) ** 2) + n * log_sigma + 0.5 * n * np.log(2 * np.pi)
    return nll

def mle_fit_model(x, y, model="linear"):
    #maximum likelihood estimation fit assuming Gaussian errors
    #returns y_hat_mle (n,), params_mle (tuple), sigma_hat (float), ll (log likelihood), nll (float)
    if model not in MODELS:
        raise ValueError(f"Model '{model}' not recognized. Choose from {list(MODELS.keys())}.")
    fn, default_p0, k = MODELS[model]

    #initial guess
    try:
        y_hat_ols, params_ols = fit_model(x, y, model=model)
        sigma0 = np.std(y - y_hat_ols) + 1e-6
        theta0 = np.array(list(params_ols) + [np.log(sigma0)], dtype=float)
    except Exception:
        #fallback
        theta0 = np.zeros(k + 1, dtype=float)

    #minimize NLL
    result = minimize(
        nll_gaussian, 
        theta0, 
        args=(x, y, fn, k), 
        method='L-BFGS-B',
        bounds=[(None, None)] * k + [(-20, 5)],  # sigma > 0
        options={'maxiter': 10000}
    )
    if not result.success:
        raise RuntimeError("MLE optimization failed: " + result.message)

    theta_hat = result.x
    params_hat = theta_hat[:k]
    sigma_hat = np.exp(theta_hat[-1])
    nll = float(result.fun)
    ll = -nll

    y_hat_mle = fn(x, *params_hat)
    return y_hat_mle, tuple(params_hat), sigma_hat, ll, nll