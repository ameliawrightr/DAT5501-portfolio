#script executing full pipeline
#reproducible - CI can run one command and get CSV + plot + results

from pathlib import Path
import json
from lab04_data_pipeline.src.line_pipeline import GenConfig, generate_data, fit_line, save_plot 

def main():
    #1. generate synthetic data
    cfg = GenConfig() #use defaults; edit if desired
    csv_p, meta_p = generate_data(cfg)
    print(f"Generated data CSV: {csv_path}, meta JSON: {meta_path}")

    #2. fit line to data
    m_est, b_est, x, y = fit_line(str(csv_p))
    print(f"Fitted line: y = {m_est:.3f}x + {b_est:.3f}")

    #3. save plot of data + true line + best-fit line
    meta = json.loads(Path(meta_p).read_text())
    plot_p = save_plot(x, y, meta["m"], meta["b"], m_est, b_est, cfg.plot_path)
    print(f"Saved line fit plot to: {plot_path}")

    Path("lab04_data_pipeline/outputs").mkdir(parents=True, exist_ok=True)
    Path("lab04_data_pipeline/outputs/results.json").write_text(json.dumps({
        "true_m": meta["m"],
        "true_b": meta["b"],
        "est_m": m_est,
        "est_b": b_est
    }, indent=2))

    print(f"CSV: {csv_p}")
    print(f"PLOT: {plot_p}")
    print(f"RESULTS: lab04_data_pipeline/outputs/results.json")

if __name__ == "__main__":
    main()
