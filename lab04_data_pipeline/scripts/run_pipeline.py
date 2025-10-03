#script executing full pipeline
#reproducible - CI can run one command and get CSV + plot + results

from pathlib import Path
import json

#import pure functions + config 
from lab04_data_pipeline.src.line_pipeline import (
    GenConfig, generate_data, fit_line, save_plot 
)


def main():
    #1. choose defaults
    cfg = GenConfig()

    #2. generate synthetic data -> write CSV + meta.json
    csv_p, meta_p = generate_data(cfg)
    print(f"Generated data CSV: {csv_p}, meta JSON: {meta_p}")

    #3. fit the model from the CSV
    m_est, b_est, x, y = fit_line(str(csv_p))
    print(f"Fitted line: y = {m_est:.3f}x + {b_est:.3f}")

    #4. load ground truth for plotting/metrics
    meta = json.loads(Path(meta_p).read_text())

    #5. save plot (scatter, true line, best-fit line)
    plot_p = save_plot(x, y, meta["m"], meta["b"], m_est, b_est, out_path=cfg.plot_path)
    print(f"Saved line fit plot to: {plot_p}")

    #6. save results JSON (true vs estimated parameters)
    #ensure output folder exists
    outputs = Path("lab04_data_pipeline/outputs")
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "results.json").write_text(json.dumps({
        "true_m": meta["m"],
        "true_b": meta["b"],
        "est_m": m_est,
        "est_b": b_est
    }, indent=2))

    #7. log paths (see where files went in local and CI logs)
    print(f"CSV: {csv_p}")
    print(f"PLOT: {plot_p}")
    print(f"RESULTS: {outputs/'results.json'}")

if __name__ == "__main__":
    main()
