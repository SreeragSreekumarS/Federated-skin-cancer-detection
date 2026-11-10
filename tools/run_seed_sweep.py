"""Run the same final profile at fixed split seed and multiple training seeds."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("centralized", "federated"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision for every seed run")
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_file():
        parser.error(f"config not found: {config}")
    for seed in args.seeds:
        run_name = f"{config.stem}_seed{seed}"
        if args.mode == "centralized":
            command = [
                sys.executable, "-m", "centralized_baseline.train",
                "--manifest", args.manifest, "--config", str(config),
                "--seed", str(seed), "--run-name", run_name,
                "--output", str(Path(args.output_root) / "centralized"),
            ]
        else:
            command = [
                sys.executable, "train.py", "--manifest", args.manifest,
                "--config", str(config), "--seed", str(seed),
                "--run-name", run_name, "--output", args.output_root,
            ]
        if args.no_amp:
            command.append("--no-amp")
        print("\nRunning:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
