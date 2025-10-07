from pathlib import Path
import json

import numpy as np
import pandas as pd

#import public API to test
from lab04_data_pipeline.src.line_pipeline import (
    GenConfig, generate_data, fit_line, save_plot,
    generate_data_quadratic, fit_quadratic_ols, generate_data_exponential, fit_exponential_mle, 
    r2_score
)    

def test_csv_saved_and_numeric(tmp_path: Path):
    #purpose: prove pipeline writes valid CSV
    # - use pytest tmp_path so tests dont interact w repo files
    # - validate numeric columns and no missing values
    cfg = GenConfig(
        csv_path=str(tmp_path/"synth.csv"),
        meta_path=str(tmp_path/"meta.json"),
        n=120, 
        noise_sigma=0.8 #moderate noise
    )

    csv_p, _meta_p = generate_data(cfg)
    assert Path(csv_p).exists(), "CSV was not saved."

    #numeric only - fail if non-numeric or NaNs
    df = pd.read_csv(csv_p).apply(pd.to_numeric, errors="raise")
    assert {"x", "y"}.issubset(df.columns), "CSV must have x, y columns."
    assert not df[["x", "y"]].isna().any().any(), "CSV contains NaN values."

def test_fit_close_to_truth(tmp_path: Path):
    #purpose: check estimated slope/intercept close to true values
    # - allow tolerances for noise + finite N
    cfg = GenConfig(
        csv_path=str(tmp_path/"synth.csv"),
        meta_path=str(tmp_path/"meta.json"),
        n=300, noise_sigma=0.6, seed=7, m=3.2, b=-1.5
    )

    csv_p, meta_p = generate_data(cfg)
    m_est, b_est, *_ = fit_line(str(csv_p))
    meta = json.loads(Path(meta_p).read_text())
    
    #tolerances chosen for noise+N
    assert np.isclose(m_est, meta["m"], atol=0.2), f"slope off: {m_est} vs {meta['m']}"
    assert np.isclose(b_est, meta["b"], atol=0.5), f"intercept off: {b_est} vs {meta['b']}"

def test_plot_saved(tmp_path: Path):
    #purpose: ensure visualisation file is created
    cfg = GenConfig(
        csv_path=str(tmp_path/"synth.csv"),
        meta_path=str(tmp_path/"meta.json"),
        plot_path=str(tmp_path/"fit.png")
    )
    
    csv_p, meta_p = generate_data(cfg)
    m_est, b_est, x, y = fit_line(str(csv_p))
    meta = json.loads(Path(meta_p).read_text())
    
    plot_p = save_plot(x, y, meta['m'], meta['b'], m_est, b_est, out_path=cfg.plot_path)
    assert Path(plot_p).exists(), "Plot was not saved."

def test_quadratic_recovery(tmp_path):
    csv_p, _ = generate_data_quadratic(
        a=0.3, b=-1.2, c=2.0, n=500,
        x_min=-2, x_max=2,
        sigma=0.3, seed=11, heteroscedastic=False,
        sigma_min=0.0, sigma_max=0.0,
        n_outliers=0, outdir=tmp_path
    )
    a_hat, b_hat, c_hat = fit_quadratic_ols(str(csv_p))
    assert abs(a_hat - 0.3) < 0.06
    assert abs(b_hat + 1.2) < 0.12
    assert abs(c_hat - 2.0) < 0.2

def test_exponential_mle_recovery(tmp_path):
    csv_p, _ = generate_data_exponential(
        a=1.8, b=0.6, n=600, x_min=0.0, x_max=4.0, sigma=0.15, seed=22, outdir=tmp_path
    )
    a_hat, b_hat = fit_exponential_mle(csv_p)
    assert abs(a_hat - 1.8) < 0.25
    assert abs(b_hat - 0.6) < 0.08

def test_gof_r2_in_range():
    y_true = np.array([0,1,2,3,4], dtype=float)
    y_pred = np.array([0,1,2,3,4], dtype=float)
    assert 0.9999 < r2_score(y_true, y_pred) <= 1.0001