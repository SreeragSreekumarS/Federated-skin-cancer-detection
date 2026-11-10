"""Report mean ± sample standard deviation for seed runs' raw and selected metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def melanoma_sensitivity(metrics: dict) -> float | None:
    per_class = metrics.get("per_class", {})
    for name, values in per_class.items():
        if "mel" in name.lower():
            return values.get("sensitivity")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Run output directory; repeat per seed")
    parser.add_argument("--output", default="seed_summary.csv")
    args = parser.parse_args()
    rows = []
    for run in args.run:
        metrics_path = Path(run) / "metrics.json"
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        for pipeline in ("raw_argmax", "selected"):
            values = report[pipeline]
            rows.append({
                "run": str(run), "pipeline": pipeline,
                "accuracy": values.get("accuracy"),
                "balanced_accuracy": values.get("balanced_accuracy"),
                "macro_f1": values.get("macro_f1"),
                "melanoma_sensitivity": melanoma_sensitivity(values),
            })
    frame = pd.DataFrame(rows)
    summary_rows = []
    for pipeline, group in frame.groupby("pipeline"):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "melanoma_sensitivity"):
            values = group[metric].dropna().to_numpy(dtype=float)
            summary_rows.append({
                "pipeline": pipeline, "metric": metric, "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
            })
    summary = pd.DataFrame(summary_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary to {output}")


if __name__ == "__main__":
    main()
