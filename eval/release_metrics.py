"""Recompute pass rates from the compact, recorded release scores."""

import json
from pathlib import Path

from eval.metrics import summarize

RESULTS = Path(__file__).resolve().parents[1] / "docs/results/f1_20261102_011103"


def main():
    rows = json.loads((RESULTS / "scored-runs.json").read_text(encoding="utf-8"))
    expected = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["metrics"]
    snapshot = Path(__file__).resolve().parents[1] / "configs/f1_multi_freeze.json"
    jobs = json.loads(snapshot.read_text(encoding="utf-8"))["jobs"]
    results = summarize(rows, jobs)
    fields = ("completed", "successes", "pass1", "pass3")
    for variant, report in results.items():
        if not report["complete"] or any(report[field] != expected[variant][field] for field in fields):
            raise ValueError(f"{variant}: recorded scores disagree with the aggregate")
        print(variant, {field: report[field] for field in fields})


if __name__ == "__main__":
    main()
