"""Summarise evaluation results across seeds.

Reads ``evaluation_results/summary.csv``, which ``evaluate_pitchdepth.py``
appends to, and groups rows tagged ``<experiment>_seed<N>_<best|final>`` (the
tags ``run_experiment.py --print-eval`` uses). For each experiment and
checkpoint it prints the mean and standard deviation of every metric over the
seeds. With ``--compare A B`` it also prints the per-seed differences B - A for
seeds both experiments share, so each comparison is paired by seed.
When a tag was evaluated more than once, the latest row is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict

METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "silog")
TAG = re.compile(r"^(?P<experiment>.+)_seed(?P<seed>\d+)_(?P<which>best|final)$")


def load(path: str) -> dict:
    """{(experiment, which): {seed: {metric: value}}}, keeping the latest row per tag."""
    rows = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in sorted(csv.DictReader(f), key=lambda r: r["timestamp"]):
            match = TAG.match(row.get("tag") or "")
            if match:
                key = (match["experiment"], match["which"])
                rows[key][int(match["seed"])] = {m: float(row[m]) for m in METRICS}
    return rows


def describe(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.6g}"
    return f"{statistics.mean(values):.6g} ± {statistics.stdev(values):.2g}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", default="evaluation_results/summary.csv")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="Also show paired per-seed differences B - A, e.g. baseline pitchdepth")
    args = p.parse_args(argv)

    rows = load(args.summary)
    if not rows:
        raise SystemExit(f"No rows tagged <experiment>_seed<N>_<best|final> in {args.summary}.")

    print("| Experiment | Checkpoint | Seeds | " + " | ".join(METRICS) + " |")
    print("| --- | --- | --- |" + " --- |" * len(METRICS))
    for (experiment, which), by_seed in sorted(rows.items()):
        cells = [describe([by_seed[s][m] for s in sorted(by_seed)]) for m in METRICS]
        print(f"| {experiment} | {which} | {len(by_seed)} | " + " | ".join(cells) + " |")

    if args.compare:
        a, b = args.compare
        print(f"\nPaired differences, {b} - {a} (negative means {b} has the lower error):\n")
        print("| Checkpoint | Seeds | " + " | ".join(METRICS) + " |")
        print("| --- | --- |" + " --- |" * len(METRICS))
        for which in ("best", "final"):
            seeds = sorted(set(rows.get((a, which), {})) & set(rows.get((b, which), {})))
            if not seeds:
                continue
            cells = [describe([rows[(b, which)][s][m] - rows[(a, which)][s][m] for s in seeds]) for m in METRICS]
            print(f"| {which} | {len(seeds)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
